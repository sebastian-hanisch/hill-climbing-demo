# Hill Climbing – eine Lieferrunde, die im ersten lokalen Optimum stecken bleibt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-hill-climbing-demo.streamlit.app/)**

Erstes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Hill Climbing (lokale Suche)** – an einem wachsenden Beispiel: ein Fahrzeug, ein Depot in der Mitte und n Kundenstopps in einem 100 × 100-km-Gebiet, eine Rundtour (TSP).
Die Suche probiert kleine Änderungen – zwei Stopps **vertauschen**, ein Stück der Tour **umdrehen (2-opt)**, ein Stück **an anderer Stelle einfügen (Or-opt)** – und behält jede, die die Tour kürzer macht. Was nach dem letzten verbessernden Zug übrig bleibt, ist ein **lokales Optimum**.

**Einordnung in die Reihe (die Kanten des Graphen):** Hill Climbing ist die **Wurzel**: eine einzige Lösung, nur Verbesserungen. Alle sechs Nachfolger setzen an derselben Schwäche an, jeder auf einem anderen Weg. Die Linie hat **keinen Konvergenzpunkt**.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)   [dieses Stück]
  ├─ Simulated Annealing      (nimmt Verschlechterungen an, Abkühlplan)                      [nicht gebaut]
  ├─ Iterated Local Search → VNS → ALNS  (stört ein Optimum; wechselt die Nachbarschaft; lernt Umbauten)   [nicht gebaut]
  ├─ Tabu Search              (Gedächtnis gegen Rückwege)                                    [nicht gebaut]
  └─ GRASP                    (randomisierte Konstruktion, viele Starts)                     [nicht gebaut]
```
(Mit dem genetischen Algorithmus der Populations-Linie ergäbe sich später ein Memetischer Algorithmus.)

Ergebnis in Kürze: **2-opt von einer zufälligen Startlösung endet bei 60 Stopps im Mittel 7.9 % über einer unteren Schranke** (Streuung von 0.9 bis 14.1 % je Lauf); ein zweiter Start endet woanders – von 100 Abstiegen liegen nur 4 % höchstens 2 % über der Schranke.
Die **Nachbarschaft** entscheidet mehr als die Startlösung: **2-opt + Or-opt** endet bei 3.7 %, jedes 2-opt-Optimum ließ sich mit Or-opt weiter verbessern (im Mittel 3.8 %). Eine **gute Startlösung** (Nächster Nachbar) spart vor allem Züge (17 statt 211); die Güte hebt sie nur mit steilstem Abstieg (4.0 % statt 7.9 %) – bei gruppierten Stopps ist sie mit erster Verbesserung sogar schlechter als Zufall.

| Frage | Ergebnis (60 gleichverteilte Stopps, ein Depot; 2-opt, zufällige Startlösung, erste Verbesserung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Startlösungen; Abstand = Prozent über der 1-Baum-Schranke, die bei gleichverteilten 60 Stopps im Mittel 0.5 % unter dem Optimum liegt) |
|---|---|
| Standardfall | ❌ **7.9 %** über der Schranke nach 211 Zügen und 74 Tausend bewerteten Nachbarn; kreuzungsfrei (die Startlösung liegt im Mittel 420 % über der Schranke; auf der Voreinstellungs-Instanz hat sie 405 Kreuzungen). Streuung je Lauf 0.9 bis 14.1 % (Standardabweichung 4.0) |
| **Nachbarschaft** | ✅ Tausch **54.5 %** (mit Nächstem Nachbarn 16.3 %; 13 Kreuzungen bleiben), 2-opt 7.9 %, Or-opt 10.4 % (2.7 Kreuzungen), **2-opt + Or-opt 3.7 %**. Die Kombination ist in allen vier Kombinationen aus Start und Regel besser als jede Nachbarschaft allein (3.4 bis 3.8 %). Nachbarn einer Tour bei 60 Stopps: Tausch und 2-opt je 1 769, Or-opt 10 614, beide 12 383 |
| **Startlösung** | ➖ Nächster Nachbar (21 % über der Schranke) statt Zufall (420 %): bei erster Verbesserung **kaum Gewinn** (7.3 % statt 7.9 %) aber 17 statt 211 Züge; mit steilstem Abstieg **4.0 % statt 7.9 %**. Bei 75 % gruppierten Stopps ist der Nächste Nachbar mit erster Verbesserung **schlechter** als Zufall (6.7 % gegen 4.1 %) |
| **Auswahlregel** | ➖ Erste gegen beste Verbesserung: bei 2-opt gleich gut (7.9 % beide), aber 211 gegen 57 Züge und 74 gegen 102 Tausend bewertete Nachbarn; bei 2-opt + Or-opt 3.7 gegen 3.8 %, 220 gegen 45 Züge, 107 gegen **573** Tausend Bewertungen |
| **Mehrfachstart** | ✅ 100 Abstiege aus zufälligen Startlösungen (dieselbe Instanz): der beste von 1 / 5 / 20 / 100 Starts liegt bei **7.9 / 4.0 / 1.8 / 1.2 %**; nur 4 % der Abstiege enden höchstens 2 %, 25 % höchstens 5 % über der Schranke. Im Mittel 99 von 100 Optima sind verschieden, sie teilen aber **74 %** ihrer Kanten mit der besten Tour: die guten Touren liegen nahe beieinander (Vorgriff auf GRASP und ILS) |
| **Nachbarschaftswechsel** | ✅ Alle 2-opt-Optima waren kein Optimum von 2-opt + Or-opt; weitere Verkürzung im Mittel 3.8 % (Vorgriff auf VNS) |
| **Größe** | ❌ Lücke wächst kaum (1.7 / 1.9 / 7.7 / 7.9 / 9.1 / 9.1 / 9.5 % bei 10 / 20 / 40 / 60 / 100 / 150 / 200 Stopps), aber Züge 12 / 45 / 109 / 211 / 403 / 672 / 991 (2- bis 5-mal n) und **bewertete Nachbarn 0.2 / 2.0 / 16 / 74 / 448 / 1 542 / 3 973 Tausend** (etwa n³, ×54 bei 3.3-facher Größe): nach jedem Zug wird alles neu bewertet, ohne Nachbarschaftslisten |
| Gruppierte Stopps | ➖ 7.9 / 8.0 / 5.7 / 4.1 / 4.4 % bei 0 / 25 / 50 / 75 / 100 % der Stopps in fünf Gruppen |
| 2-opt + Or-opt, Nächster Nachbar, steilster Abstieg | ✅ 0.05 / 1.1 / 3.4 / 4.4 / 5.3 / 5.1 % bei 20 / 40 / 60 / 100 / 150 / 200 Stopps nach 3 / 6 / 9 / 18 / 27 / 31 Zügen – die beste Kombination, aber auch sie wird mit der Größe schlechter |
| Untere Schranke | ⚠️ Die 1-Baum-Schranke (Held-Karp, Subgradientenverfahren) liegt bei gleichverteilten 60 Stopps im Mittel **0.5 %** unter dem Optimum (CP-SAT), bei gruppierten **1.1 %** (ein Einzelfall 3.7 %): der angezeigte Abstand überschätzt die echte Lücke um diesen Betrag |

## Was die Demo zeigt

1. **Hill Climbing in Aktion** (Schritt-Slider + Abspielen): **Instanz** (Stopps und Depot) → **Startlösung** (Länge, Abstand zur Schranke, Kreuzungen) → **Nachbarschaft** (der erste Zug mit entfernten und neuen Kanten, dazu die Verteilung der Längenänderung aller Nachbarn) →
   **Abstieg** (Zug-Regler und ▶️ Züge abspielen: die Tour nach jedem Zug, dazu die Kurve der Tourlänge über die Züge und die Schranke) → **Lokales Optimum** (vorher / nachher, Kontrolle: keiner der Nachbarn der Endtour ist kürzer).
2. **Was die Suche gefunden hat:** Länge, Abstand zur Schranke, Züge (nach Art), bewertete Nachbarn und Rechenzeit; Urteil (`near_optimal` unter 3 % über der Schranke → `crossings` (Optimum der Nachbarschaft, aber Kreuzungen) → `stuck`), Detailtabelle.
3. **📐 Sweeps** über Stopps, Anteil in Gruppen, Nachbarschaft, Startlösung und Auswahlregel (feste Instanzen ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** alle 16 Kombinationen aus Nachbarschaft, Startlösung und Regel; **Mehrfachstart** mit 100 Abstiegen (Histogramm, bester Stand nach k Starts, verschiedene Optima, gemeinsame Kanten); **2-opt-Optima mit Or-opt weitersuchen**; **Skalierung** von 20 bis 200 Stopps für zwei Verfahren.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (nur verbessernde Züge, die Startlösung ist gleichgültig, die Nachbarschaft ist groß genug, alle Nachbarn zu bewerten ist billig, die Schranke ist das Optimum).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen (0–100 %), **Nachbarschaft** (Tausch / 2-opt / Or-opt / 2-opt + Or-opt), **Startlösung** (Zufällig / Nächster Nachbar), **Auswahlregel** (erste / beste Verbesserung), Seed der Instanz (+ 🎲), Seed der Startlösung (+ 🎲; bei Nächstem Nachbarn ausgeblendet, der Wert bleibt erhalten).

## Messwerte der Presets (Instanz-Seed 35, Start-Seed 0; sie prüfen sich mit weiten Bändern selbst)

| Preset | Abstand zur Schranke | Züge | bewertete Nachbarn | Kreuzungen (Start → Ende) | Urteil |
|---|---|---|---|---|---|
| Standardfall (Voreinstellung) | 6.8 % | 202 | 59 486 | 405 → 0 | stuck |
| Nächster Nachbar als Start | 5.4 % | 26 | 14 297 | 11 → 0 | stuck |
| Nächster Nachbar + steilster Abstieg | 0.8 % | 23 | 42 456 | 11 → 0 | near_optimal |
| Nur Tausch | 63.5 % | 178 | 45 450 | 405 → 12 | crossings |
| Nur Or-opt | 10.9 % | 323 | 64 042 | 405 → 1 | crossings |
| 2-opt + Or-opt | 3.5 % | 210 | 90 696 | 405 → 0 | stuck |
| Große Instanz (200 Stopps) | 10.7 % | 1 026 | 5 989 352 | 4 122 → 0 | stuck |

Die Presets zeigen einzelne Instanzen; die Mittelwerte über fünf Instanzen stehen in der Tabelle oben und in den Hilfetexten der Regler.

## Modell und Verfahren

- **Instanz** (`hc_scenario.py`): Depot in der Mitte, n Stopps gleichverteilt oder zu einem einstellbaren Anteil in fünf Gruppen (Mittelpunkte mindestens 12 km vom Rand, Streuung 6 km), euklidische Entfernungen, alles durch den Seed festgelegt.
- **Suche** (`hc_algorithm.py`, numpy von Grund auf): Tour als Permutation, zyklisch. Die Längenänderung **jedes** Nachbarn wird aus wenigen Kanten berechnet und je Suchdurchgang als N × N-Matrix bewertet (2-opt, Tausch, Or-opt je Segmentlänge 1–3, beide Richtungen; die Zahl der *logisch* bewerteten Nachbarn wird gezählt, bei erster Verbesserung nur bis zum ersten kürzenden). Erste Verbesserung = erster kürzender Nachbar in fester Reihenfolge, beste = kleinstes Delta.
  Nach jedem Zug beginnt die Suche von vorn (keine Nachbarschaftslisten, keine Don't-Look-Bits – bewusst, damit die Kosten sichtbar bleiben). **Kreuzungen** werden über Orientierungstests gezählt.
- **Untere Schranke:** 1-Baum (Prim auf den Knoten ohne das Depot + die zwei billigsten Depotkanten), **Held-Karp-Subgradientenverfahren** (300 Schritte, Polyak-Schrittweite mit der Länge eines guten lokalen Optimums als Ziel).
- **Auswertung** (`hc_evaluation.py`): Kennzahlen, Sweeps und Vergleichstabellen über feste Instanzen (je drei Startlösungen), Mehrfachstart, Optima-Hierarchie (2-opt → 2-opt + Or-opt), Skalierung, Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "Zufallsstart + 2-opt endet deutlich über dem Optimum, vielleicht zweistellig" – **falsch in der Größenordnung**: 7.9 % über der Schranke, 2-opt + Or-opt 3.7 %; nur der Tausch ist wirklich schlecht (54 %). (2) "Der Nächste Nachbar als Start ist besser und braucht weniger Züge" – **halb falsch**: weniger Züge ja (17 statt 211), bessere Güte nur mit steilstem Abstieg (4.0 % statt 7.9 %); mit erster Verbesserung 7.3 % statt 7.9 %, bei gruppierten Stopps sogar **schlechter** als Zufall (6.7 % gegen 4.1 %).
  (3) "Erste gegen beste Verbesserung: gleiche Güte, die beste braucht mehr Bewertungen je Zug" – bestätigt, dazu **fünfmal so viele Bewertungen insgesamt** bei 2-opt + Or-opt (573 gegen 107 Tausend). (4) "Die lokalen Optima teilen sich viele Kanten mit der besten Tour" – **bestätigt** (74 %; eine zufällige Tour teilt im Erwartungswert nur 2/(N−1) ≈ 3 % ihrer Kanten mit einer beliebigen anderen). (5) "Züge wachsen mit n bis n log n" – Züge etwa 2- bis 5-mal n, **bewertete Nachbarn etwa mit n³** (das war die eigentliche Überraschung der Skalierung). (6) "Ein 2-opt-Optimum ist kein Or-opt-Optimum" – bestätigt (alle Fälle, 3.8 % weitere Verkürzung).
  Nicht vorhergesagt: die **Lücke wächst mit der Größe kaum** (9.5 % bei 200 Stopps), und dass die **Schranke** so eng ist (0.5 % im Mittel) – ohne Löser genügt sie als Maßstab.
- **Kein exakter Löser in der Demo:** das Optimum (CP-SAT, `AddCircuit`) steht nur in den Tests als Kontrolle; die App misst gegen die Schranke.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Wie sich die Verfahren auf Straßennetzen oder mit Nebenbedingungen verhalten, zeigt diese Demo nicht. Die Kosten (Bewertungen nach jedem Zug neu, n³) sind die der einfachen Implementierung; in der Praxis senken Nachbarschaftslisten und Don't-Look-Bits sie stark.
  Zahlen für Or-opt bei 200 Stopps sind langsam (einige Sekunden je Lauf).

## Verifikation

- **Algorithmus:** Längenänderung **jedes** Kandidaten (Tausch, 2-opt, Or-opt beider Richtungen, Segmentlängen 1–3, mit Umlauf über das Array-Ende) gegen die neu gemessene Tourlänge; Nachbarschaften gegen eine **unabhängige Aufzählung** mit expliziten Listenoperationen (Minimum und Anzahl der Nachbarn, Handformel N(N−3)/2 und Or-opt-Zählung); erste Verbesserung = erster kürzender Nachbar in Zeilenreihenfolge;
  Abstieg **strikt monoton**, endet in einem lokalen Optimum (per Vollprüfung mit der unabhängigen Aufzählung), deterministisch; ein 2-opt-Optimum ist **kreuzungsfrei**, Tausch und Or-opt lassen Kreuzungen stehen; Kreuzungszählung an Handinstanzen; Nächster Nachbar an Handinstanz; **1-Baum-Schranke**: Handinstanz (Quadrat), ≤ Optimum (Brute-Force n = 8, CP-SAT n = 25) und höchstens 3 % darunter.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Vergleichstabelle, Mehrfachstart, Nachbarschaftswechsel, Skalierung, Grenzen-Tabelle, Schrankenlücke gegen CP-SAT; jeweils Mittel über die festen Sweep-Instanzen; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung);
  alle 7 Presets über mehrere Instanzen und Startlösungen in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Zug-Regler, ▶️ Abspielen und ▶️ Züge abspielen ohne doppelte Schlüssel, ausgeblendeter Start-Seed, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Vergleich, Mehrfachstart, Nachbarschaftswechsel, Skalierung), 🚧 Grenzen, Mathe |
| `hc_algorithm.py` | Nachbarschaften und ihre Längenänderung, Abstieg, Kreuzungen, 1-Baum-Schranke, Mehrfachstart |
| `hc_scenario.py`, `hc_constants.py` | Instanzen (gleichverteilt, in Gruppen); Konstanten, Presets |
| `hc_evaluation.py` | Kennzahlen, Analyse, Urteil, Sweeps, Vergleichstabellen, Skalierung |
| `hc_presets.py`, `hc_visualization.py` | Permalink/Presets (ausgeblendeter Start-Seed), Plotly-Figuren (achsengesperrt) |
| `tests/` | Algorithmus (Brute-Force-Aufzählung, CP-SAT), Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
