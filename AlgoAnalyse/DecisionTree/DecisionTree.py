from decision_tree_pipeline_class import DiabetesDecisionTreePipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


general_config = {
    "data_path": "Datensatz/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    "target_col": "Diabetes",
    "test_size": 0.2,
    "random_state": 42,
    "max_depth": 5,
    "min_samples_split": 20,
    "criterion": "gini",
    "verbose": True,
    "plots_enable": False,
    "state_of_test" : "Final_Decision" # "Analysis"
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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.train_test_split()

    pipeline.init_model()
    pipeline.fit()

    if general_config["plots_enable"]:
        pipeline.plot_tree(title="Baseline (alle Features)")

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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    top_features = ["HighBP", "BMI", "GenHlth", "Age", "Income", "PhysHlth"]
    pipeline.set_top_features(top_features)

    pipeline.train_test_split()
    pipeline.init_model()
    pipeline.fit()

    if general_config["plots_enable"]:
        pipeline.plot_tree(title="Top-Features Baum")

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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    pipeline.init_model(max_depth=5, min_samples_split=20)

    scores = pipeline.cross_validate(cv=5)

    if general_config["plots_enable"]:
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(scores) + 1), scores)
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Cross-Validation Scores (5-Fold)")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        plt.show()

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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion="entropy",   # ntropy wählen
        verbose = general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.train_test_split()

    pipeline.init_model()
    pipeline.fit()
    if general_config["plots_enable"]:
        pipeline.plot_tree(title="Decision Tree – Entropy")

    acc = pipeline.evaluate()
    return float(acc)

def scenario_grid_search_tree(return_all=False):
    """
    Führt eine Grid‑Search für den Decision Tree durch.

    Es werden verschiedene Kombinationen von Baum‑Parametern getestet
    (z.B.. max_depth, min_samples_split, min_samples_leaf, criterion).
    Jede Kombination wird per 5‑facher Cross‑Validation bewertet.

    Ziel:
        Die Parameter finden, mit denen der Decision Tree
        im Durchschnitt die höchste Accuracy erreicht.

    Rückgabe:
        Die beste gefundene Accuracy (float) oder
        (score, params, model, feature_importances) wenn return_all=True.
    """
    pipeline = DiabetesDecisionTreePipeline(
        data_path=general_config["data_path"],
        random_state=general_config["random_state"],
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
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

    results = pd.DataFrame(grid.cv_results_)

    if general_config["plots_enable"]:
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=results,
            x="param_max_depth",
            y="mean_test_score",
            marker="o"
        )
        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.title("Grid Search: Einfluss von max_depth")
        plt.grid(True)
        plt.show()

    best_model = grid.best_estimator_
    best_score = grid.best_score_
    best_params = grid.best_params_

    # ---------------------------------------------------------
    # NEU: Feature Importances aus dem besten Modell extrahieren
    # ---------------------------------------------------------
    feature_importances = pd.Series(
        best_model.feature_importances_,
        index=pipeline.X.columns
    ).sort_values(ascending=False)

    pipeline._log("\nWichtigste Features (Grid Search Modell):")
    pipeline._log(feature_importances)

    # ---------------------------------------------------------
    # Rückgabe erweitern
    # ---------------------------------------------------------
    if return_all:
        return best_score, best_params, best_model, feature_importances

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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
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

    if general_config["plots_enable"]:
        alphas = [a for a, s in scores]
        accs = [s for a, s in scores]

        plt.figure(figsize=(8, 4))
        plt.plot(alphas, accs, marker="o")
        plt.xlabel("ccp_alpha")
        plt.ylabel("Accuracy")
        plt.title("Pruning: Accuracy über verschiedene Alpha-Werte")
        plt.grid(True)
        plt.show()

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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
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
    scores = []  #Liste für Plot

    # 3) Schnelle CV (cv=3 statt cv=5)
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            random_state=pipeline.random_state,
            ccp_alpha=alpha
        )
        score = cross_val_score(tree, pipeline.X, pipeline.y, cv=3, scoring="accuracy").mean()
        scores.append((alpha, score))  # <-- für Plot merken

        if score > best_score:
            best_score = score
            best_alpha = alpha

    # Plot: Accuracy über Alpha
    if general_config["plots_enable"]:
        alphas = [a for a, s in scores]
        accs = [s for a, s in scores]

        plt.figure(figsize=(8, 4))
        plt.plot(alphas, accs, marker="o")
        plt.xlabel("ccp_alpha")
        plt.ylabel("Accuracy")
        plt.title("Pruning (schnell): Accuracy über verschiedene Alpha-Werte")
        plt.grid(True)
        plt.show()

    # Logging
    pipeline._log(f"Bestes Alpha: {best_alpha}")
    pipeline._log(f"Bester Score: {best_score}")

    return float(best_score)



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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
        verbose=general_config["verbose"]
    )

    pipeline.load_data()
    pipeline.split_features_target()

    pipeline.init_model(ccp_alpha=best_alpha)

    scores = cross_val_score(pipeline.model, pipeline.X, pipeline.y, cv=5, scoring="accuracy")

    if general_config["plots_enable"]:
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(scores) + 1), scores)
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title(f"CV Scores mit best_alpha={best_alpha:.5f}")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        plt.show()


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
        max_depth=general_config["max_depth"],
        min_samples_split=general_config["min_samples_split"],
        criterion=general_config["criterion"],
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


def build_pipeline_from_best_params(best_params):
    pipeline = DiabetesDecisionTreePipeline(
        data_path = general_config["data_path"],
        random_state = general_config["random_state"],
        max_depth = best_params.get("max_depth"),
        min_samples_split = best_params.get("min_samples_split"),
        min_samples_leaf = best_params.get("min_samples_leaf"),
        criterion=best_params.get("criterion"),
        verbose=True
    )

    pipeline.load_data()
    pipeline.split_features_target()
    pipeline.train_test_split()

    # Überschreibe mit wichtigen Parametern
    pipeline.init_model(
        max_depth = best_params.get("max_depth"),
        min_samples_split=best_params.get("min_samples_split"),
        min_samples_leaf=best_params.get("min_samples_leaf", 1),
        criterion=best_params.get("criterion")
    )

    pipeline.fit()
    return pipeline

def print_final_summary(score, params, importances):
    print("\n" + "="*60)
    print("                FINALE MODELL-ZUSAMMENFASSUNG")
    print("="*60)

    print("\nBeste Accuracy (CV-basiert):")
    print(f"  {score:.4f}")

    print("\nBeste Hyperparameter:")
    for k, v in params.items():
        print(f"  {k:20s}: {v}")

    print("\nWichtigste Features:")
    print(importances.head(10).to_string())

    print("\n" + "="*60)
    print("                ENDE DER ZUSAMMENFASSUNG")
    print("="*60)

if __name__ == "__main__":
    if general_config["state_of_test"] == "Analysis":
        test_config ={
            #"all_features": scenario_all_features,
            #"top_features": scenario_top_features,
            #"cross_validation": scenario_cross_validation,
            #"entropy": scenario_entropy,
            "grid_search_tree": scenario_grid_search_tree,  # Dauert länger
            # "pruning": scenario_pruning, # Dauert sehr lange
            #"pruning_fast": scenario_pruning_fast,
            #"pruning_fast_with_best_alpha": scenario_pruning_fast_with_best_alpha
        }
        #choose_best_test_variant(test_config)

    '''
    Analyse Phase 1 abgeschlossen
    Nach der Ausführung aller Scenarien, hat sich gezeigt, 
    das grid_search_tree die besten ergebnisse liefert.
    Mit diesem könnte man jetzt weitere analysen durchführen.
    Dazu fehlt leider die Zeit.
    
    Daher wird final nur der Algorithmus der die besten Ergebnisse liefert
    hier nochmal seperat ausgeführt und das Ergebnis doklumentiert
    '''
    '''
    if general_config["state_of_test"] == "Final_Decision":
        #  Grid Search erneut ausführen, aber diesmal mit return_all=True um die besten Parameter zu bekopenn
        score, params, model, importances = scenario_grid_search_tree(return_all=True)

        print("\nBeste Parameter:", params)
        print("Beste Accuracy:", score)

        # Pipeline mit besten Parametern bauen
        pipeline_best = build_pipeline_from_best_params(params)

        # Evaluieren
        pipeline_best.evaluate()

        #  Baum plotten
        pipeline_best.plot_tree(title="Best Model (Grid Search)")

        # Feature Importance
        importances.head().plot(kind="barh", figsize=(8, 6))
        plt.title("Feature Wichtigkeit (Grid Search Modell)")
        plt.show()
        '''
    if general_config["state_of_test"] == "Final_Decision":

        score, params, model, importances = scenario_grid_search_tree(return_all=True)

        # Pipeline mit besten Parametern bauen
        pipeline_best = build_pipeline_from_best_params(params)

        # Evaluieren
        pipeline_best.evaluate()

        # Plots
        if general_config["plots_enable"]:
            pipeline_best.plot_tree(title="Best Model (Grid Search)")
            importances.head().plot(kind="barh", figsize=(8, 6))
            plt.title("Feature Wichtigkeit (Grid Search Modell)")
            plt.show()

        # Finale Zusammenfassung
        print_final_summary(score, params, importances)


