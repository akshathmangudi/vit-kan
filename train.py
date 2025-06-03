import os
import torch
import argparse
import logging
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm, trange
from torchvision import transforms
from model import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from utils import calculate_metrics, save_metrics, setup_logging

def main(train_loader, test_loader, args):
    device = torch.device(args.device)
    
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("No GPUs available, using CPU")

    # Create model
    model = VisionTransformer(
        (3, 32, 32), 
        n_patches=args.n_patches,
        n_blocks=args.n_blocks, 
        d_hidden=args.d_hidden, 
        n_heads=args.n_heads, 
        out_d=100,
        type=args.model_type
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        # Adjust batch size for multiple GPUs
        effective_batch_size = args.batch_size * torch.cuda.device_count()
        logging.info(f"Effective batch size: {effective_batch_size}")
    
    model = model.to(device)

    # Log model parameters
    if hasattr(model, 'module'):
        # DataParallel wraps the model, so access the underlying module
        total_params = sum(p.numel() for p in model.module.parameters())
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Use AdamW with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps,
        eta_min=args.learning_rate * 0.01
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    metrics_log_filename = setup_logging(args.log_dir)
    
    # Initialize AMP scaler if using mixed precision
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    best_test_acc = 0.0
    step = 0

    for epoch in trange(args.epochs, desc="train"):
        train_loss = 0.0
        y_true_train, y_pred_train, y_pred_proba_train = [], [], []

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Mixed precision training if available
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                y_hat = model(x)
                loss = criterion(y_hat, y)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            scheduler.step()
            step += 1

            train_loss += loss.detach().cpu().item() / len(train_loader)

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_train.extend(torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy())
            
            # Log learning rate periodically
            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                logging.debug(f"Epoch {epoch+1}, Batch {batch_idx}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(y_true_train, y_pred_train, y_pred_proba_train)

        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Train Accuracy: {accuracy:.4f}")
        logging.info(f"  Train Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"  Train F1 Score: {f1:.4f}")
        logging.info(f"  Train ROC AUC: {roc_auc:.4f}")
        logging.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Validation/Testing every few epochs or at the end
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                y_true_test, y_pred_test, y_pred_proba_test = [], [], []

                for batch in tqdm(test_loader, desc="Testing", leave=False):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            y_hat = model(x)
                            loss = criterion(y_hat, y)
                    else:
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                    
                    test_loss += loss.detach().cpu().item() / len(test_loader)

                    y_true_test.extend(y.cpu().numpy())
                    y_pred_test.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
                    y_pred_proba_test.extend(torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy())

                accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(y_true_test, y_pred_test, y_pred_proba_test)

                logging.info("Test Results:")
                logging.info(f"  Test Loss: {test_loss:.4f}")
                logging.info(f"  Test Accuracy: {accuracy:.4f}")
                logging.info(f"  Test Balanced Accuracy: {balanced_accuracy:.4f}")
                logging.info(f"  Test F1 Score: {f1:.4f}")
                logging.info(f"  Test ROC AUC: {roc_auc:.4f}")

                # Save best model
                if accuracy > best_test_acc:
                    best_test_acc = accuracy
                    if args.save_model:
                        # Handle DataParallel when saving
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'test_accuracy': accuracy,
                            'args': args
                        }, os.path.join(args.log_dir, 'best_model.pth'))
                        logging.info(f"New best model saved with accuracy: {accuracy:.4f}")

                save_metrics(metrics_log_filename, epoch + 1, "Test", test_loss, accuracy, balanced_accuracy, f1, roc_auc, flag=1)

        # Save training metrics
        if epoch == args.epochs - 1:
            save_metrics(metrics_log_filename, epoch + 1, "Train", train_loss, accuracy, balanced_accuracy, f1, roc_auc, flag=0)

    logging.info(f"Training completed. Best test accuracy: {best_test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Vision Transformer on CIFAR-100')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for training')
    parser.add_argument('--model-type', type=str, default='vanilla', help='variant to run')
    parser.add_argument('--n-blocks', type=int, default=8, help='number of transformer blocks')
    parser.add_argument('--d-hidden', type=int, default=384, help='hidden dimension of transformer block')
    parser.add_argument('--n-heads', type=int, default=6, help='number of attention heads')
    parser.add_argument('--n-patches', type=int, default=8, help='number of patches (8x8 = 64 patches for 32x32 images)')
    parser.add_argument('--log-dir', type=str, default='logs', help='directory to store logs')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing factor')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='number of warmup epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='gradient clipping norm (0 to disable)')
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--eval-freq', type=int, default=5, help='evaluation frequency (epochs)')
    parser.add_argument('--save-model', action='store_true', help='save best model')
    args = parser.parse_args()

    # Improved CIFAR-100 Transformations
    cifar_train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        transforms.RandomErasing(p=0.1)
    ])

    cifar_test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load CIFAR-100 Dataset
    train_dataset = CIFAR100(root='./cifar100', train=True, download=True, transform=cifar_train_transforms)
    test_dataset = CIFAR100(root='./cifar100', train=False, download=True, transform=cifar_test_transforms)

    # For DataParallel, keep num_workers reasonable
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    main(train_loader=train_loader, test_loader=test_loader, args=args)