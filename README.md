# 🩺 Diabetes Prediction – Decision Tree Analyse  
Ein Machine‑Learning‑Projekt zur Vorhersage von Diabetes und Analyse relevanter Einflussfaktoren

## 🎯 Projektziel

Ziel des Projekts ist es zu untersuchen, ob sich Diabetes zuverlässig vorhersagen lässt und welche Faktoren dabei die größte Rolle spielen.  
Dazu wurde ein Decision‑Tree‑Classifier in mehreren Szenarien trainiert, optimiert und bewertet.  
Die Analyse basiert auf öffentlich verfügbaren Datensätzen (z. B. Kaggle Diabetes Health Indicators).

---

## 📂 Datengrundlage

Verwendeter Datensatz:

- **Diabetes Health Indicators Dataset**  
  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Ziel der Aufgabenstellung war es außerdem, den Code so zu abstrahieren, dass beide Datensätze mit derselben Pipeline verarbeitet werden können.

---

## 🧠 Vorgehensweise & Szenarien

Die Analyse wurde in mehreren strukturierten Szenarien durchgeführt, um die Modellqualität systematisch zu verbessern und die wichtigsten Einflussfaktoren zu identifizieren.

---

### 1️⃣ **Baseline‑Szenario (alle Features)**  
Ein Decision Tree wurde mit allen verfügbaren Merkmalen trainiert.  
Zweck dieses Szenarios:

- Grundgenauigkeit bestimmen  
- Overfitting erkennen  
- Ausgangspunkt für spätere Optimierungen schaffen  

---

### 2️⃣ **Top‑Features‑Szenario**  
Es wurde untersucht, ob eine reduzierte Feature‑Menge bessere Ergebnisse liefert.  
Typische Top‑Features waren u. a.:

- HighBP  
- BMI  
- Age  

Ziele:

- Modell vereinfachen  
- Interpretierbarkeit erhöhen  
- Overfitting reduzieren  

---

### 3️⃣ **Cross‑Validation‑Szenario**  
Um zufallsbedingte Verzerrungen zu vermeiden, wurde eine k‑fold Cross‑Validation durchgeführt.

Nutzen:

- Stabilere Accuracy  
- Robustere Bewertung  
- Vergleichbarkeit der Szenarien  

---

### 4️⃣ **Entropy‑Szenario (Split‑Kriterium)**  
Vergleich der Kriterien:

- `gini` (Standard)  
- `entropy`  

Ziel:

- Prüfen, ob ein alternatives Kriterium präzisere Splits erzeugt  
- Einfluss auf Overfitting und Genauigkeit bewerten  

---

### 5️⃣ **GridSearch‑Szenario (Hyperparameter‑Optimierung)**  
Systematische Suche nach optimalen Parametern:

- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  
- `criterion`  

Ziel:

- Maximale Modellgenauigkeit  
- Bestes Bias‑Variance‑Verhältnis  

Dieses Szenario bildet den Kern der Optimierungsphase.

---

### 6️⃣ **Pruning‑Szenarien (klassisch & schnell)**  
Pruning reduziert Overfitting durch Vereinfachung des Baums.

Zwei Varianten:

- **klassisches Pruning**: viele `ccp_alpha`‑Werte, sehr gründlich  
- **schnelles Pruning**: reduzierte Werte + 3‑fold CV  

Ziel:

- kompakterer Baum  
- stabilere Vorhersagen  

---

### 7️⃣ **Best‑Alpha‑Szenario**  
Der beste zuvor gefundene `ccp_alpha`‑Wert wurde erneut getestet und per Cross‑Validation validiert.

Ziel:

- finale Stabilitätsprüfung  
- Grundlage für die endgültige Modellpipeline  

---

## 🧩 Architektur & Pipeline

Für das Projekt wurde eine modulare **Diabetes‑Pipeline‑Klasse** entwickelt:

- flexibel konfigurierbare Parameter  
- wiederverwendbare Methoden  
- Unterstützung aller Szenarien  
- klare Trennung von Datenvorbereitung, Training und Evaluation  

Die Pipeline ermöglicht es, verschiedene Datensätze mit minimalen Anpassungen zu verarbeiten.

---

## 📊 Ergebnisse (Kurzüberblick)

- Feature‑Selektion verbessert die Interpretierbarkeit und reduziert Overfitting  
- Cross‑Validation liefert stabilere Ergebnisse als ein einzelner Train/Test‑Split  
- Das Split‑Kriterium (`gini` vs. `entropy`) beeinflusst die Modellstruktur, aber nicht immer die Accuracy  
- GridSearch liefert die größten Performance‑Sprünge  
- Pruning führt zu kompakteren und robusteren Bäumen  
- Der finale Baum basiert auf dem optimalen `ccp_alpha`‑Wert  

---

## 🧪 Erweiterungen laut Aufgabenstellung

- Durchführung von **zwei Projekten mit unterschiedlichen Datensätzen**  
- Überprüfung, ob die Pipeline abstrahiert werden kann  
- Durchführung und Bewertung eines **Clustering‑Verfahrens** (z. B. K‑Means)  
- Vergleich der Ergebnisse zwischen den Datensätzen  

---

## 👥 Autor

**Nils Bauch**  
Machine Learning & Data Science  

