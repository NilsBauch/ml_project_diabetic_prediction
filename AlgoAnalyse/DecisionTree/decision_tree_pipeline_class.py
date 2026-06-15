import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class DiabetesDecisionTreePipeline:
    def __init__(
        self,
        data_path,
        target_col="Diabetes",
        test_size=0.2,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        ccp_alpha=0.0,
        verbose=True
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose

        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.top_features = None

    # -------------------------------------------------------------
    # 1) Daten laden
    # -------------------------------------------------------------
    def load_data(self, info=True, describe=True, head=True):
        self.df = pd.read_csv(self.data_path)

        if head:
            self._log("Erste Zeilen:")
            self._log(self.df.head())
        if info:
            self._log("\nInfo:")
            self._log(self.df.info())
        if describe:
            self._log("\nDescribe:")
            self._log(self.df.describe())

    # -------------------------------------------------------------
    # 2) Features/Target trennen
    # -------------------------------------------------------------
    def split_features_target(self):
        self.X = self.df.drop(self.target_col, axis=1)
        self.y = self.df[self.target_col]

    # -------------------------------------------------------------
    # 3) Train/Test Split
    # -------------------------------------------------------------
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    # -------------------------------------------------------------
    # 4) Modell initialisieren & trainieren
    # -------------------------------------------------------------
    def init_model(self, max_depth=None, min_samples_split=None, min_samples_leaf=None, criterion=None, ccp_alpha=None):
        if max_depth is not None:
            self.max_depth = max_depth
        if min_samples_split is not None:
            self.min_samples_split = min_samples_split
        if min_samples_leaf is not None:
            self.min_samples_leaf = min_samples_leaf
        if criterion is not None:
            self.criterion = criterion
        if ccp_alpha is not None:
            self.ccp_alpha = ccp_alpha  # <-- überschreibt Klassenattribut

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha
        )

    def fit(self):
        if self.model is None:
            self.init_model()
        self.model.fit(self.X_train, self.y_train)

    # -------------------------------------------------------------
    # 5) Feature Importances
    # -------------------------------------------------------------
    def show_feature_importances(self, top_n=None):
        importances = pd.Series(self.model.feature_importances_, index=self.X.columns)
        importances_sorted = importances.sort_values(ascending=False)

        if top_n:
            importances_sorted = importances_sorted.head(top_n)

        self._log(f"\nFeature Importances:")
        self._log(importances_sorted)
        return importances_sorted

    # -------------------------------------------------------------
    # 6) Evaluation
    # -------------------------------------------------------------
    def evaluate(self, show_report=True):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)

        self._log(f"\nTest Accuracy: {acc}")

        if show_report:
            self._log(f"\nClassification Report:")
            self._log(classification_report(self.y_test, y_pred))

        self._log("\nOverfitting Check:")
        self._log(f"Train Accuracy: {self.model.score(self.X_train, self.y_train)}")
        self._log(f"Test Accuracy: {self.model.score(self.X_test, self.y_test)}")

        return acc

    # -------------------------------------------------------------
    # 7) Baum visualisieren
    # -------------------------------------------------------------
    def plot_tree(self, max_depth=5, figsize=(15, 10), title=None):
        plt.figure(figsize=figsize)

        plot_tree(
            self.model,
            feature_names=self.X.columns,
            filled=True,
            max_depth=max_depth,  # <-- HIER wird die Tiefe begrenzt
        )

        if title is None:
            title = f"Decision Tree (max_depth={max_depth if max_depth else 'full'})"

        plt.title(title)
        plt.show()

    # -------------------------------------------------------------
    # 8) Top-Features setzen & neu trainieren
    # -------------------------------------------------------------
    def set_top_features(self, feature_list):
        self.top_features = feature_list
        self.X = self.df[self.top_features]

    def retrain_with_top_features(self):
        self.train_test_split()
        self.init_model()
        self.fit()

    # -------------------------------------------------------------
    # 9) Grid Search Light (max_depth × min_samples_split)
    # -------------------------------------------------------------
    '''
    def grid_search(self, depth_list, split_list):
        results = []

        for d in depth_list:
            for s in split_list:
                self.init_model(max_depth=d, min_samples_split=s)
                self.fit()
                acc = self.evaluate(show_report=False)

                results.append({
                    "max_depth": d,
                    "min_samples_split": s,
                    "accuracy": acc
                })

        df_results = pd.DataFrame(results)
        self._log(f"\nGrid Search Ergebnisse:")
        self._log(df_results.sort_values("accuracy", ascending=False))

        best = df_results.sort_values("accuracy", ascending=False).iloc[0]
        self._log("\nBeste Parameterkombination:")
        self._log(best)

        # Modell auf beste Parameter setzen
        self.init_model(
            max_depth=int(best["max_depth"]),
            min_samples_split=int(best["min_samples_split"])
        )
        self.fit()

        return df_results
    '''
    # -------------------------------------------------------------
    # 10) Cross Value Score für stabilere Evaluation
    # -------------------------------------------------------------
    def cross_validate(self, cv=5):
        if self.model is None:
            self.init_model()

        scores = cross_val_score(
            self.model,
            self.X,
            self.y,
            cv=cv,
            scoring="accuracy"
        )

        self._log(f"{cv}-Fold Cross Validation Accuracy: {scores.mean():.4f}")
        self._log(f"Einzelergebnisse: {scores}")
        return scores

    def _log(self, msg):
        if not self.verbose:
            return

        # DataFrame → String
        if hasattr(msg, "to_string"):
            print(msg.to_string())
        else:
            print(msg)
