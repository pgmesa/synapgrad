
import pkbar
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)

MIN_MODE = 'min'
MAX_MODE = 'max'

class Evaluator:
    
    def __init__(self, epoch_callback=None, step_callback=None, accuracy=True) -> None:
        self.accuracy_bool = accuracy
        self.epoch_callback = epoch_callback
        self.step_callback = step_callback
        self.y_true = np.array([], dtype=np.uint0)
        self.y_pred = np.array([], dtype=np.uint0)
        self.binary_low_boundary = 0
        self.binary_high_boundary = 1
        
    def reset(self):
        self.y_true = np.array([], dtype=np.uint0)
        self.y_pred = np.array([], dtype=np.uint0)
    
    def step(self, labels, outputs, prefix=None) -> list:
        """ Calculates custom metrics for each step

        Args:
            labels (torch.Tensor): tensor with labels (0 or 1) in each coordinate
            outputs (torch.Tensor): tensor with labels not rounded, as the model returned
            prefix (_type_, optional): Whether to add a prefix to the metrics, 'val' for example. Defaults to None.

        Returns:
            tuple: [('metric_name', metric_val), ...]
        """
        
        labels_numpy = labels.squeeze().detach().numpy()
        outputs_numpy = outputs.squeeze().detach().numpy()
        
        if len(labels_numpy.shape) > 1:
            # Multi-class
            y_true = np.argmax(labels_numpy, axis=1)
            y_pred = np.argmax(outputs_numpy, axis=1)
        else:
            # Binary-class
            y_true = labels_numpy
            low = self.binary_low_boundary; high = self.binary_high_boundary
            y_pred = np.where(outputs_numpy > (high+low)/2, high, low)
            
        y_true = y_true.astype(np.uint0)
        y_pred = y_pred.astype(np.uint0)
        
        self.y_true = np.concatenate([self.y_true, y_true], axis=0, dtype=np.uint0)
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0, dtype=np.uint0)
        
        metrics = self.__compute(y_true, y_pred, prefix, callback=self.step_callback)
        
        return metrics
    
    def basic_accuracy_callback(self, y_true:np.ndarray, y_pred:np.ndarray) -> list:
        train_accuracy = ((y_true == y_pred).sum() / len(y_true))
        
        return [("accuracy", train_accuracy)]
    
        
    def compute(self, prefix=None) -> list:
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        self.reset()
        
        return self.__compute(y_true, y_pred, prefix, callback=self.epoch_callback)
    
    def __compute(self, y_true, y_pred, prefix, callback=None) -> list:
        metrics = []
        if self.accuracy_bool:
            metrics += self.basic_accuracy_callback(y_true, y_pred)
        if callback is not None:
            metrics += callback(y_true, y_pred)
        
        if prefix is not None:
            metrics_with_prefix = []
            for (m, v) in metrics:
                metrics_with_prefix.append((f"{prefix}_{m}", v))
            metrics = metrics_with_prefix
            
        return metrics
            
             
class Trainer:
    
    def __init__(self, model, engine) -> None:
        self.model = model
        self.engine = engine
        
    def compile(self, loss_fn, optimizer_fn, evaluator:Evaluator=None):
        self.criterion = loss_fn
        self.optimizer = optimizer_fn
        self.evaluator = evaluator
        
    
    def fit(self, train_loader, epochs, validation_loader=None, monitor='val_loss',
                on_train_epoch:callable=None, on_validation_epoch:callable=None):
        self.epochs = epochs
        self.history = {}
        
        def record_metrics(dictionary, metrics:list[tuple]):
            for (k,v) in metrics:
                assert type(v) is float or type(v) is np.float_, f"Metric value recorded is not 'float' but '{type(v)}'"
                if dictionary.get(k, False): dictionary[k].append(v)
                else: dictionary[k] = [v]
                
        if monitor == 'val_loss':
            monitor = 'val_loss' if validation_loader is not None else 'loss'
            
        for epoch in range(epochs):
            ############ TRAIN ############
            self.model.train()
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=epochs, width=32, always_stateful=False)
            if on_train_epoch is not None: on_train_epoch(self.model, train_loader)
            train_metrics = self.__train(train_loader, kbar)
            record_metrics(self.history, train_metrics)
            ########### Validation ##########
            if validation_loader is not None:
                if on_validation_epoch is not None: on_validation_epoch(self.model, validation_loader)
                val_metrics = self.__validate(validation_loader)
                # ================= Update Validation Metrics Info ==================
                record_metrics(self.history, val_metrics)
                # ===================================================================
            else: val_metrics = []
            kbar.add(1, values=train_metrics + val_metrics)
            
        return self.history
    
    def __train(self, train_loader, kbar:pkbar.Kbar) -> tuple:
        """ Train model for one epoch """
        self.model.train()
        epoch_train_loss = 0
        for i, data in enumerate(train_loader):
            *inputs, labels = data
            # ========================== Predict ===========================
            outputs = self.model(*inputs).squeeze(dim=1)
            # ======================== Step Metrics ========================
            train_loss = self.criterion(outputs, labels)
            metrics = self.evaluator.step(labels, outputs) if self.evaluator != None else []
            # ========= Clean, Update Gradients and Update Weights =========
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            # ================= Update Train Metrics Info ==================
            epoch_train_loss += train_loss.item()
            train_metrics = [("loss", train_loss.item())] + metrics
            kbar.update(i, values=train_metrics)
            # ==============================================================
        train_loss = epoch_train_loss / (i + 1)
        epoch_metrics = self.evaluator.compute() if self.evaluator != None else []
       
        return [("loss", train_loss)] + epoch_metrics        
    
    def __validate(self, validation_loader) -> tuple:
        """ Validate model with validation data """
        self.model.eval()
        total_val_loss = 0
        with self.engine.no_grad():
            for i, data in enumerate(validation_loader):
                *inputs, labels = data
                # ========================== Predict ==========================
                outputs = self.model(*inputs).squeeze(dim=1)
                # ========================== Metrics ==========================
                val_loss = self.criterion(outputs, labels)
                total_val_loss += val_loss.item()
                self.evaluator.step(labels, outputs, prefix='val') if self.evaluator != None else []
            
            val_loss = total_val_loss / (i + 1)
            val_metrics = self.evaluator.compute(prefix='val') if self.evaluator != None else []
        
        return [('val_loss', val_loss)] + val_metrics
        
    def plot(self, metrics:list=['loss'], figsize=(15,5), style=False, ylim:tuple=None):
        if style: plt.style.use("ggplot")
        
        plt.figure(figsize=figsize)
        def plot_history_metric(metric):
            plt.plot(self.history[f"{metric}"])
            val_metric = f"val_{metric}"
            if val_metric in self.history: plt.plot(self.history[val_metric])
            plt.title('Train History')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend(['train', 'validation'], loc='upper left')
        
        for i, m in enumerate(metrics):
            plt.subplot(1,len(metrics), i+1)
            plot_history_metric(m)
        plt.show()
    
    # def test(self, test_loader, report=True, classes=None) -> np.ndarray:
    #     # we can now evaluate the network on the test set
    #     print("[INFO] Testing network...")
    #     # set the model in evaluation mode
    #     self.model.eval()
    #     if self.model.training: raise Exception("Model is in training mode")
    #     preds = []; y_test = []
    #     with torch.no_grad():
    #         for data in test_loader:
    #             *inputs, labels = data
    #             for l in labels.cpu().numpy():
    #                 y_test.append(l)
    #             inputs = self.__to_device(inputs)
    #             pred = self.model(*inputs).squeeze(dim=1)
    #             pred = pred.cpu()
    #             for p in pred:
    #                 preds.append(p.numpy())

    #     y_test = np.array(y_test)
    #     y_pred = np.array(preds)
        
    #     if len(y_test.shape) > 1:
    #         # Multi-class
    #         y_test = np.argmax(y_test, axis=1)
    #         y_pred = np.argmax(y_pred, axis=1)
    #     else:
    #         # Binary-class
    #         y_pred = np.where(y_pred > 0.5, 1, 0)
        
    #     if report:
    #         print("Accuracy:", accuracy_score(y_test, y_pred))
    #         print("ROC_AUC:", roc_auc_score(y_test, y_pred))
    #         print(confusion_matrix(y_test, y_pred))
    #         print(classification_report(y_test, y_pred, target_names=classes))
        
    #     return y_pred, y_test
                