import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Any, Dict

def get_device() -> torch.device:
    
    """ Get the available device (GPU or CPU) """
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model_name: str, num_labels: int, device: torch.device) -> Tuple[Any, Any]:
   
    """ Initialize the tokenizer and model """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return tokenizer, model

def calculate_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    
    """ Calculate the accuracy of predictions """
    
    _, predicted = torch.max(preds, dim=1)
    return (predicted == labels).sum().item()

def train_epoch(model: Any, data_loader: Any, optimizer: Any, device: torch.device) -> Tuple[float, float]:
    
    """ Train the model for one epoch """
    
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        correct_predictions += calculate_accuracy(logits, batch['labels'])
        total_samples += batch['labels'].size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def eval_model(model: Any, data_loader: Any, device: torch.device) -> Tuple[float, float]:
    
    """ Evaluate the model """

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            correct_predictions += calculate_accuracy(logits, batch['labels'])
            total_samples += batch['labels'].size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def train_and_validate_model(
    model_name: str,
    train_loader: Any,
    val_loader: Any,
    learning_rate: float,
    num_labels: int,
    epochs: int = 4,
) -> Any:
    
    """ Train and validate the BERT model """

    save_path = f"{model_name}_finetuned_model_lr_{learning_rate}"
    print(f"Saving model to: {save_path}")

    device = get_device()
    tokenizer, model = initialize_model(model_name, num_labels, device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):

        """ For each epoch ... """

        print(f"Epoch {epoch}/{epochs}")
        
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = eval_model(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    model.push_to_hub(save_path)
    tokenizer.push_to_hub(save_path)

    return model