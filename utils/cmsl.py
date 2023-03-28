import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances

class CMSL(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf = None, times=1, mode = "single", random_state=None):
        self.base_clf = base_clf
        self.times = times
        self.mode = mode
        self.random_state = random_state
        self.random = np.random.RandomState(seed=self.random_state)
        self.normalizer = StandardScaler()
        
    def fit(self, X_list, y):
        self.Xs = []
        self.y = y
        self.classes = np.unique(y)
        
        _, counts = np.unique(self.y, return_counts=True)
        class_weights = np.array([1-(weight/np.sum(counts)) for weight in counts])
        sample_weights = class_weights[self.y]
        
        # Get all modalities
        for X_modality in X_list:
            X_modality = self.normalizer.fit_transform(X_modality)
            self.Xs.append(X_modality)
        
        # CLustering all modalities
        self.clfs = []
        
        y_indx = np.arange(0, self.y.shape[0], 1)
        seed_idxs = []
        for i in self.classes:
            class_idxs = y_indx[self.y==i]
            # print(class_idxs)
            chosen = self.random.choice(class_idxs, self.times)
            seed_idxs.append(chosen)
        seed_idxs = np.array(seed_idxs).ravel()
        
        if self.mode == "seed":
            for X_modality in self.Xs:
                self.clfs.append(clone(self.base_clf).fit(X_modality[seed_idxs], self.y[seed_idxs], sample_weight=class_weights[self.y[seed_idxs]]))
        
        elif self.mode == "single":
            for X_modality in self.Xs:
                modality_centroids = X_modality[seed_idxs]
                modality_distances = pairwise_distances(X_modality, modality_centroids)
                closest_centroids = np.argsort(modality_distances, axis=1)[:, 0]
                
                propagated_y = self.y[seed_idxs][closest_centroids]
                
                self.clfs.append(clone(self.base_clf).fit(X_modality, propagated_y, sample_weight=class_weights[propagated_y]))
                
        elif self.mode == "cross":
            # Get both distances
            distances = []
            propagated_labels = []
            for X_modality in self.Xs:
                modality_centroids = X_modality[seed_idxs]
                modality_distances = pairwise_distances(X_modality, modality_centroids)
                closest_centroids = np.argsort(modality_distances, axis=1)[:, 0]
                
                distance_to_closest = np.sort(modality_distances, axis=1)[:, 0]
                
                scaler = MinMaxScaler()
                # distance_to_closest = scaler.fit_transform(distance_to_closest.reshape(1, -1)).ravel()
                
                propagated_y = self.y[seed_idxs][closest_centroids]
                distances.append(distance_to_closest)
                propagated_labels.append(propagated_y)
                
            distances = np.array(distances)
            propagated_labels = np.array(propagated_labels)
            # print(distances)
            print(propagated_labels)
            different = propagated_labels[0].shape[0] - np.where(np.equal(propagated_labels[0], propagated_labels[1]))[0].shape[0]
            print(different)
            exit()
            
            closest_in_both = np.argmin(distances, axis=0)
            cross_labels = propagated_labels.T[np.arange(len(propagated_labels.T)), closest_in_both]
            
            for X_modality in self.Xs:
                self.clfs.append(clone(self.base_clf).fit(X_modality, cross_labels, sample_weight=class_weights[cross_labels]))
        
        else:
            print("No such mode!")
        
        return self
    
    def predict(self, X_list):
        X_preds = []
        for X_modality in X_list:
            X_modality = self.normalizer.fit_transform(X_modality)
            X_preds.append(X_modality)
        
        preds = [self.clfs[modality_id].predict(X_modality) for modality_id, X_modality in enumerate(X_preds)]
        return tuple(preds)
    

class mmBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf = None, mode = "single", normalized = False):
        self.base_clf = base_clf
        self.mode = mode
        self.normalized = normalized
        self.normalizer = StandardScaler()
        
    def fit(self, X_list, y):
        self.Xs = []
        self.y = y
        self.classes = np.unique(y)
        
        _, counts = np.unique(self.y, return_counts=True)
        class_weights = np.array([1-(weight/np.sum(counts)) for weight in counts])
        sample_weights = class_weights[self.y]
        
        # Get all modalities
        if self.normalized:
            for X_modality in X_list:
                X_modality = self.normalizer.fit_transform(X_modality)
                self.Xs.append(X_modality)
        else:
            self.Xs = X_list
        
        # Fit classifier for each modality
        self.clfs = []
        for X_modality in self.Xs:
            self.clfs.append(clone(self.base_clf).fit(X_modality, self.y, sample_weight=sample_weights))
        
        return self
    
    def predict(self, X_list):
        X_preds = []
        if self.normalized:
            for X_modality in X_list:
                X_modality = self.normalizer.fit_transform(X_modality)
                X_preds.append(X_modality)
        else:
            X_preds = X_list
            
        preds = [self.clfs[modality_id].predict(X_modality) for modality_id, X_modality in enumerate(X_preds)]
        return tuple(preds)
    
    def predict_combined(self, X_list):
        X_preds = []
        for X_modality in X_list:
            X_modality = self.normalizer.fit_transform(X_modality)
            X_preds.append(X_modality)
            
        esm = np.array([self.clfs[modality_id].predict_proba(X_modality) for modality_id, X_modality in enumerate(X_preds)])
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)
        return prediction