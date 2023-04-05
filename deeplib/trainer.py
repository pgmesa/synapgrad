
import pkbar
import numpy as np
import matplotlib.pyplot as plt

from deeplib.losses import Loss

class Trainer:
    
    def __init__(self, model) -> None:
        self.model = model
    
    def fit(self, train_loader, epochs, criterion:Loss, optimizer, validation_split:float=None, validation_loader=None, show_pbar=True):
        self.history = {}
        
        def record_metrics(dictionary, metrics:list[tuple]):
            for (k,v) in metrics:
                if dictionary.get(k, False): dictionary[k].append(v)
                else: dictionary[k] = [v]
        
        for epoch in range(epochs):
            ############ TRAIN ############
            if show_pbar:
                kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=epochs, width=32, always_stateful=False)
            epoch_train_loss = 0; epoch_train_acc = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                # ========================== Predict ===========================
                outputs = self.model(inputs)
                outputs = np.squeeze(outputs)
                # ======================== Step Metrics ========================
                train_loss = criterion(outputs, labels)
                train_loss = np.mean(train_loss)
                
                # -- Metrics
                if len(labels.shape) > 1:
                    # Multi-class
                    y_true = np.argmax(labels, axis=1)
                    y_pred = np.argmax(outputs, axis=1)
                else:
                    # Binary-class
                    y_true = labels
                    y_pred = np.where(outputs > 0.5, 1, 0)
                    
                y_true = y_true.astype(np.uint0)
                y_pred = y_pred.astype(np.uint0)
                
                train_accuracy = ((y_true == y_pred).sum() / len(y_true))
                # ========= Calculate gradients, Backpropagate error and Update Weights =========
                optimizer.step(criterion.backward())
                # ================= Update Train Metrics Info ==================
                epoch_train_loss += train_loss; epoch_train_acc += train_accuracy
                train_metrics = [("loss", train_loss), ("accuracy", train_accuracy)]
                if show_pbar: kbar.update(i, values=train_metrics)
                # ==============================================================
            train_loss = epoch_train_loss / (i + 1)
            train_accuracy = epoch_train_acc / (i + 1)
            epoch_train_metrics = [("loss", train_loss), ("accuracy", train_accuracy)]
            record_metrics(self.history, epoch_train_metrics)
            ########### Validation ##########
            if validation_loader is not None:
                total_val_loss = 0; total_val_accuracy = 0
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    # ========================== Predict ==========================
                    outputs = self.model(inputs)
                    outputs = np.squeeze(outputs)
                    rounded_outputs = (outputs > 0.5).float()
                    # ========================== Metrics ==========================
                    val_loss = criterion(outputs, labels.float())
                    # -----val_accuracy = (labels == rounded_outputs).sum() / len(labels)
                    total_val_loss += val_loss
                    total_val_accuracy += val_accuracy
                val_loss = total_val_loss / (i + 1)
                val_accuracy = total_val_accuracy / (i + 1)
                # ================= Update Validation Metrics Info ==================
                val_metrics = [("val_loss", val_loss), ("val_accuracy", val_accuracy)]
                record_metrics(self.history, val_metrics)
                if show_pbar: kbar.add(1, values=val_metrics)
                # ==============================================================
            
            if not show_pbar:
                string = f"EPOCH {epoch+1}/{epochs} loss: {train_loss} accuracy: {train_accuracy}"
                if validation_loader is not None:
                    val_info = f" val_loss: {val_loss} val_accuracy: {val_accuracy}"
                    string += val_info 
                print(string+"\n")
        return self.history
        
    def plot(self, metrics:list=['loss'], figsize=(15,5), style=False):
        if style: plt.style.use("ggplot")
        
        plt.figure(figsize=figsize)
        def plot_history_metric(metric):
            plt.plot(self.history[f"{metric}"])
            val_metric = f"val_{metric}"
            if val_metric in self.history:
                plt.plot(self.history[val_metric])
            plt.title('Train History')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper left')
        
        for i, m in enumerate(metrics):
            plt.subplot(1,len(metrics), i+1)
            plot_history_metric(m)
        plt.show()
    
    def test(self, test_loader, report=True, classes=None):
        # we can now evaluate the network on the test set
        print("[INFO] Testing network...")
        # set the model in evaluation mode
        # turn off autograd for testing evaluation
        y_test = []; X_test = []
        with torch.no_grad():
            # initialize a list to store our predictions
            preds = []
            # loop over the test set
            for (x, y) in test_loader:
                for l in y.cpu().numpy():
                    y_test.append(l)
                for t in np.squeeze(x.numpy()):
                    X_test.append(t)
                # send the input to the device
                x = x.to(self.device)
                # make the predictions and add them to the list
                pred = self.model(x).squeeze(dim=1)
                pred = pred.clone().detach()
                pred = (pred.numpy() > 0.5).astype(np.uint0)
                for p in pred:
                    preds.append(p)

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        y_pred = np.array(preds)
        
        if report:
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("ROC_AUC:", roc_auc_score(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, target_names=classes))
        
        return y_pred, y_test, X_test