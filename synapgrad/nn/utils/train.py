
import pkbar
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)

MIN_MODE = 'min'
MAX_MODE = 'max'

class Evaluator:
    
    # output of model is scalar as well as labels (0 or 1) ex: [0,1,0,1,0]
    BINARY = 'binary' 
    # output of model is vector but labels are scalars with values [0, n_class) ex: [0,3,9,2,1] (10 classes)
    MULTI_CLASS = 'multi-class'
    # output of model is vector and labels are one-hot encoded ex: [[0,0,1] [0,1,0]]
    CATEGORICAL = 'categorical'
    
    
    def __init__(self, epoch_callback=None, step_callback=None, accuracy=True, mode=MULTI_CLASS) -> None:
        self.accuracy_bool = accuracy
        self.epoch_callback = epoch_callback
        self.step_callback = step_callback
        self.y_true = np.array([], dtype=np.int16)
        self.y_pred = np.array([], dtype=np.int16)
        self.mode = mode
        
    def reset(self):
        self.y_true = np.array([], dtype=np.int16)
        self.y_pred = np.array([], dtype=np.int16)
    
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
        
        
        if self.mode == self.BINARY:
            y_pred = np.where(outputs_numpy > 0.5, 1, 0)
            y_true = labels_numpy
        elif self.mode == self.MULTI_CLASS:
            y_pred = np.argmax(outputs_numpy, axis=1)
            y_true = labels_numpy
        elif self.mode == self.CATEGORICAL:
            y_pred = np.argmax(outputs_numpy, axis=1)
            y_true = np.argmax(labels_numpy, axis=1)
        else:
            raise RuntimeError(f"Evaluator: mode '{self.mode}' is not valid")
            
            
        y_true = y_true.astype(np.int16)
        y_pred = y_pred.astype(np.int16)
        
        self.y_true = np.concatenate([self.y_true, y_true], axis=0, dtype=np.int16)
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0, dtype=np.int16)
        
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
    
    def report(self, y_pred, y_true, classes=None):
        
        auc = None
        if self.mode == self.BINARY:
            y_pred = np.where(y_pred > 0.5, 1, 0)
            # Compute AUC
            auc = roc_auc_score(y_true, y_pred)
        elif self.mode == self.MULTI_CLASS:
            y_pred = np.argmax(y_pred, axis=1)
        elif self.mode == self.CATEGORICAL:
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            raise RuntimeError(f"Evaluator: mode '{self.mode}' is not valid")
        
        print("Accuracy:", accuracy_score(y_true, y_pred))
        if auc is not None: print("ROC_AUC:", auc)
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=classes))
            
             
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
                assert np.issubdtype(type(v), np.floating), f"Metric value recorded is not 'float' but '{type(v)}'"
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
        
    def plot(self, metrics:list=['loss'], figsize=(15,5), style=False, ylim:tuple=(0,1)):
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
    
    def test(self, test_loader) -> tuple[np.ndarray, np.ndarray]:
        print("[INFO] Testing network...")
        # set the model in evaluation mode
        self.model.eval()
        if self.model.training: raise Exception("Model is in training mode")
        preds = []; y_true = []
        with self.engine.no_grad():
            for data in test_loader:
                *inputs, labels = data
                for l in labels.numpy():
                    y_true.append(l)
                pred = self.model(*inputs).squeeze(dim=1)
                for p in pred:
                    preds.append(p.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(preds)
        
        return y_pred, y_true
                