from decision_tree_pipeline_class import DiabetesDecisionTreePipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


general_config = {
    "data_path": "Datensatz/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    "target_col": "Diabetes",
    "test_size": 0.2,
    "random_state": 42,
    "max_depth": None,
    "min_samples_split": 2,
    "criterion": "gini",
    "verbose": True
}

def scenario_all_features():
    """
        Szenario: Baseline-Modell mit allen verfügbaren Features.

        Grund
        -----------------------
        Szenario dient als Ausgangspunkt (Baseline), um zu verstehen,
        wie gut ein Decision Tree mit dem vollständigen Datensatz funktioniert bzw. performt.
        Es zeigt:
        - die maximale Informationsbasis (keine Feature-Reduktion)
        - wie stark Overfitting auftreten kann
        - wie gut der Baum ohne Optimierungen funktioniert

        Nutzen:
        -------
        Dieses Szenario dient dazu, spätere Varianten (Top-Features,
        Cross-Validation, Pruning, Grid-Search)  vergleichen zu können.
        """
    data_path = "Datensatz/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

    pipeline = DiabetesDecisionTreePipeline(
        data_path= general_config["data_path"],
        test_size=general_config["test_size"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.train_test_split()

    pipeline.init_model()
    pipeline.fit()

    acc = pipeline.evaluate()
    #acc = result["accuracy"]

    #print("\n[test_all_features] Accuracy:", acc)
    return float(acc)

def scenario_top_features():
    """
    Szenario: Modell mit ausgewählten Top-Features.

    Grund
    -----------------------
    Das Szenario testet, ob ein Decision Tree mit reduzierten
    Features besser ist. Die ausgewählten Features basieren
    auf:
    - Feature-Importances
    - Domain-Wissen (z. B. HighBP, BMI, Age)
    - Korrelationen (z. B. Income vs. Education)

    Nutzen:
    -------
    Das Szenario zeigt:
    -  weniger Features --> weniger Overfitting führen
    - ob die Accuracy stabil bleibt oder sogar steigt
    """
    data_path = "Datensatz/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

    pipeline = DiabetesDecisionTreePipeline(
        data_path= general_config["data_path"],
        test_size=general_config["test_size"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    top_features = ["HighBP", "BMI", "GenHlth", "Age", "Income", "PhysHlth"]
    pipeline.set_top_features(top_features)

    pipeline.train_test_split()
    pipeline.init_model()
    pipeline.fit()

    acc = pipeline.evaluate()
    #acc = result["accuracy"]

    #print("\n[test_top_features] Accuracy:", acc)
    return float(acc)

def scenario_cross_validation():
    """
    Szenario: Modellbewertung mit Cross-Validation (CV).

    Grund
    -----------------------
    Train/Test-Split kann zufallsabhängig sein und liefern ggf.
    instabile Ergebnisse.
    Cross-Validation löst dieses Problem, indem
    das Modell mehrfach auf verschiedenen Splits trainiert und getestet
    wird.

    Nutzen:
    -------
    Szenario liefert:
    -  robustere Accuracy-Schätzung
    - geringere Varianz der Ergebnisse
    - eine realistischere Einschätzung des Modells auf unbekannten Daten

    CV ist besonders wichtig bei:
    - kleinen oder unausgewogenen Datensätzen (unausgewogen liegt hier nicht vor)
    - Feature-Engineering-Vergleichen
    - Hyperparameter-Tuning
    """
    pipeline = DiabetesDecisionTreePipeline(
        data_path= general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    pipeline.init_model(max_depth=5, min_samples_split=20)

    scores = pipeline.cross_validate(cv=5)
    mean_score = scores.mean()

    #print("\n[test_cross_validation] CV Accuracy:", mean_score)
    return float(mean_score)

def scenario_entropy():
    """
    Szenario: Decision Tree mit dem Split-Kriterium entropy anststatt gini.

    Grund
    -----------------------
    Standardmäßig verwendet scikit-learn das Gini-Kriterium (schnell und
    robust).
    Das Entropy-Kriterium basiert  auf dem
    Informationsgewinn (Information Gain) und kann bei bestimmten Datensätzen
    präzisere Splits erzeugen.

    Dieses Szenario testet:
    - ob 'entropy' auf dem Datensatz bessere Ergebnisse liefert
    - ob die Splits informativer sind (höherer Informationsgewinn)
    - ob Overfitting reduziert oder verstärkt wird
    - wie sich die Wahl des Split-Kriteriums auf die Accuracy auswirkt

    Nutzen:
    -------
    Das Szenario dient dazu, zu verstehen, ob das Standardkriterium
    (Gini) wirklich optimal ist oder ob Entropy eine bessere Alternative darstellt.
    Es ist ein zentraler Bestandteil jeder Entscheidungsbaum-Analyse.
    """

    data_path = "Datensatz/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

    pipeline = DiabetesDecisionTreePipeline(
        data_path= general_config["data_path"],
        test_size=general_config["test_size"],
        random_state=general_config["random_state"],
        criterion="entropy",   # ntropy wählen
        verbose = general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.train_test_split()

    pipeline.init_model()
    pipeline.fit()

    acc = pipeline.evaluate()
    return float(acc)

def scenario_grid_search_tree():
    """
    Führt eine Grid‑Search für den Decision Tree durch.

    Es werden verschiedene Kombinationen von Baum‑Parametern getestet
    (z.B.. max_depth, min_samples_split, min_samples_leaf, criterion).
    Jede Kombination wird per 5‑facher Cross‑Validation bewertet.

    Ziel:
        Die Parameter finden, mit denen der Decision Tree
        im Durchschnitt die höchste Accuracy erreicht.

    Rückgabe:
        Die beste gefundene Accuracy (float).
    """
    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.init_model()  # Basis-Tree

    param_grid = {
        "max_depth": [3, 5, 7, 9, None],
        "min_samples_split": [2, 10, 20, 50],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"]
    }

    grid = GridSearchCV(
        pipeline.model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(pipeline.X, pipeline.y)

    best_model = grid.best_estimator_
    best_score = grid.best_score_

    pipeline._log(f"Best params: {grid.best_params_}")
    return float(best_score)

def scenario_pruning():
    """
    Testet verschiedene ccp_alpha‑Werte (Pruning‑Stärken) für den Decision Tree.

    Für jeden Alpha‑Wert wird ein Baum trainiert und per 5‑facher
    Cross‑Validation bewertet. Der Alpha‑Wert mit der höchsten Accuracy
    wird ausgewählt.

    Rückgabe:
        Die beste gefundene Accuracy (float).
    """
    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    # Erstmal ohne Pruning trainieren, um ccp_alphas zu bekommen
    temp_tree = DecisionTreeClassifier(random_state=pipeline.random_state)
    path = temp_tree.cost_complexity_pruning_path(pipeline.X, pipeline.y)
    ccp_alphas = path.ccp_alphas

    scores = []
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            random_state=pipeline.random_state,
            ccp_alpha=alpha
        )
        score = cross_val_score(tree, pipeline.X, pipeline.y, cv=5, scoring="accuracy").mean()
        scores.append((alpha, score))

    best_alpha, best_score = max(scores, key=lambda x: x[1])
    return float(best_score)



def scenario_pruning_fast():
    """
    Szenario: Cost-Complexity-Pruning (schnelle Version)

    Grund
    -----------------------
    Pruning reduziert Overfitting durch vereinfachung des Baums
    Normalerweise testet hunderte ccp_alpha-Werte und ist sehr langsam.
    Diese optimierte Version reduziert die Anzahl der getesteten Alphas
    und verwendet 3-Fold-CV statt 5-Fold, um die Laufzeit massiv zu verkürzen.

    Nutzen:
    -------
    -  schneller (10–20x)
    - stabile Accuracy-Schätzung
    - findet den besten Pruning-Parameter (ccp_alpha)
    """

    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    # 1) Pfad bestimmen (einmaliger Baum)
    temp_tree = DecisionTreeClassifier(random_state=pipeline.random_state)
    path = temp_tree.cost_complexity_pruning_path(pipeline.X, pipeline.y)
    ccp_alphas = path.ccp_alphas

    # 2) Alpha-Reduktion (20 Werte statt 200)
    ccp_alphas = np.linspace(ccp_alphas.min(), ccp_alphas.max(), 20)

    best_score = -1
    best_alpha = None

    # 3) Schnelle CV (cv=3 statt cv=5)
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            random_state=pipeline.random_state,
            ccp_alpha=alpha
        )
        score = cross_val_score(tree, pipeline.X, pipeline.y, cv=3, scoring="accuracy").mean()

        if score > best_score:
            best_score = score
            best_alpha = alpha

    # Optional: Logging
    pipeline._log(f"Bestes Alpha: {best_alpha}")
    pipeline._log(f"Bester Score: {best_score}")

    # 4) Rückgabe des besten Scores
    return best_score

def scenario_pruning_fast_with_best_alpha():
    """
    Nutzt den zuvor berechneten besten ccp_alpha‑Wert und bewertet
    den Decision Tree damit per 5‑facher Cross‑Validation.

    Ablauf:
    1. best_alpha wird über find_best_alpha() bestimmt.
    2. Ein neues Pipeline‑Objekt wird erstellt.
    3. Das Modell wird mit diesem Alpha initialisiert.
    4. Die Accuracy wird per Cross‑Validation berechnet.

    Rückgabe:
        Durchschnittliche Accuracy als float.
    """
    best_alpha = find_best_alpha()

    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    pipeline.init_model(ccp_alpha=best_alpha)

    scores = cross_val_score(pipeline.model, pipeline.X, pipeline.y, cv=5, scoring="accuracy")

    return float(scores.mean())




def find_best_alpha():
    """
    Ermittelt den besten ccp_alpha‑Wert für das Pruning.

    Dafür werden mehrere Alpha‑Werte getestet. Für jeden Wert wird ein
    Decision Tree mit Cross‑Validation bewertet. Der Alpha‑Wert mit der
    höchsten Accuracy wird ausgewählt.

    Rückgabe:
        Der beste gefundene ccp_alpha‑Wert (float).
    """
    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    temp_tree = DecisionTreeClassifier(random_state=pipeline.random_state)
    path = temp_tree.cost_complexity_pruning_path(pipeline.X, pipeline.y)
    ccp_alphas = np.linspace(path.ccp_alphas.min(), path.ccp_alphas.max(), 20)

    best_alpha = None
    best_score = -1

    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=pipeline.random_state, ccp_alpha=alpha)
        score = cross_val_score(tree, pipeline.X, pipeline.y, cv=3, scoring="accuracy").mean()

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def choose_best_test_variant(test_config):
    def choose_best_test_variant():
        """
        Führt mehrere Testszenarien nacheinander aus und vergleicht deren Accuracy.

        Für jedes eingetragene Szenario wird:
        - das Szenario gestartet
        - die erzielte Accuracy gespeichert
        - das Ergebnis ausgegeben

        Anschließend wird das Szenario mit der höchsten Accuracy bestimmt
        und zusammen mit allen Ergebnissen angezeigt.
        """

    tests = test_config

    results = {}

    print("\n--- STARTE ALLE SZENARIEN ---")

    for name, func in tests.items():
        print("\n----------------------------------------------------------")
        print(f"\n>>> Starte Szenario: {name}")
        score = func()
        results[name] = score
        print(f"Ergebnis {name}: {score}")
        print("\n----------------------------------------------------------")

    # Beste Variante bestimmen
    best = max(results, key=results.get)

    print("\n--- SZENARIEN-ÜBERSICHT ---")
    for name, score in results.items():
        print(f"{name:20s} : {score:.4f}")

    print("\n--- BESTES SZENARIO ---")
    print(f"Variante : {best}")
    print(f"Accuracy : {results[best]:.4f}")

    return best, results




if __name__ == "__main__":
    test_config ={
        # "all_features": scenario_all_features,
        # "top_features": scenario_top_features,
        "cross_validation": scenario_cross_validation,
        # "entropy": scenario_entropy,
        "grid_search_tree": scenario_grid_search_tree,  # Dauert länger
        # "pruning": scenario_pruning, # Dauert sehr lange
        # "pruning_fast": scenario_pruning_fast,
        "pruning_fast_with_best_alpha": scenario_pruning_fast_with_best_alpha
    }
    choose_best_test_variant(test_config)
