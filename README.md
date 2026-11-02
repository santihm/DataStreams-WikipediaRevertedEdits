# Which Wikipedia Edits Will Be Reverted? A Streaming Assistant for Patrollers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santihm/DataStreams-WikipediaRevertedEdits/blob/main/wikipedia_revert_prediction.ipynb)

Group project for **Machine Learning in Time Series and Data Streams** (Master's in Applied Artificial Intelligence, UC3M), by Javier Aceituno Monja, Santiago Herron Mulet and Javier Ramírez Zarzoso.

> The notebook is written in Spanish. This README summarizes it in English.

## The problem

English Wikipedia receives about 500 edits per hour from anonymous users, and roughly one in four ends up reverted, mostly because of vandalism. Volunteer patrollers cannot review them all, and they see them in arrival order.

The idea: estimate, at the moment each edit arrives, how likely it is to be reverted, and push the most suspicious ones to the top of the queue. This needs a classifier that **learns online**: one pass over the data, no stored history, and **labels that arrive late**, because nobody knows an edit was bad until someone reverts it.

The metric that matters is practical: *if a patroller can only look at 10% of the edits, how many of the bad ones do they find?*

## Data

Anonymous edits to articles of English Wikipedia, 8 to 17 October 2026, downloaded from the Wikimedia `recentchanges` API. An edit is labelled as reverted if the API tagged it `mw-reverted`. The notebook downloads the dataset automatically from the group repository, [javiacey16/proyecto-series-temporales](https://github.com/javiacey16/proyecto-series-temporales), which also contains the collection script. Wikipedia content is available under CC BY-SA.

## Method

- **Prequential evaluation** (predict first, then learn), with the first 3 days used to tune hyperparameters and the rest for evaluation.
- **No leakage.** A 6-hour delay is simulated on the labels and on the page history the model can see.
- **Features** computed when the edit arrives: text (edit summary plus page title), size change, hour of day, summary flags (empty, section edit, "undo"), recent activity of the page, and the history of known labels of that page.
- **Seven incremental models** built with [river](https://riverml.xyz): text with words, text with characters, numeric logistic regression, an Adaptive Random Forest (ARF) on numeric features, a linear stacking, an ARF stacking, and an ARF stacking with two text models.
- **Extra analyses:** label-delay sensitivity, time and memory, clustering (KMeans and DBSTREAM), data drift and concept drift detection (ADWIN, DDM, Page-Hinkley, KSWIN), a simulated patrol and a live demo on Wikimedia EventStreams.

## Results

| Model | AUC | Precision @10% | Recall @10% |
|---|---|---|---|
| Random | 0.501 | 0.251 | 0.102 |
| Size rule | 0.543 | 0.298 | 0.122 |
| Text (words) | 0.655 | 0.444 | 0.181 |
| Text (characters) | 0.648 | 0.409 | 0.167 |
| Numeric (logistic) | 0.665 | **0.542** | **0.221** |
| Numeric ARF | 0.636 | 0.457 | 0.186 |
| **Stacking (linear)** | **0.682** | 0.526 | 0.214 |
| Stacking ARF | 0.658 | 0.468 | 0.191 |
| Stacking ARF, 2 text models | 0.662 | 0.471 | 0.192 |

- **Reviewing 10% of the edits finds about 21% of the reverted ones** with the best models, against 10% at random and 12% with a size-based rule.
- **Label delay matters a lot.** With immediate labels the AUC is 0.812. With a 6-hour delay it is 0.682, and with 24 hours it is 0.668. Ignoring the delay gives an overly optimistic evaluation.
- **Page history is the most useful feature group.** Removing it lowers the logistic AUC from 0.665 to 0.613. Text adds a smaller gain (+0.016 for logistic, +0.022 for ARF).
- **Trees did not win here.** ARF is 0.030 AUC below logistic regression on numeric features. Possible reasons are the short period, the delayed labels and the fact that ARF was not tuned, but this is not proven.
- **Speed is not a problem.** Every model processes at least about 195 edits per second, while edits arrive at roughly 0.14 per second.
- **Edits that mention "undo" or "revert"** have a revert rate near 48%, about twice the average.
- **Drift.** Numeric distributions are stable and the vocabulary turns over, with no sustained drop in daily AUC. Continuing to learn beat freezing the model or resetting it periodically, although in a single 7-day run. Against an injected shift, ADWIN reacted within minutes while KSWIN took almost 16 hours.
- **Live demo.** The model ranked incoming edits from EventStreams in real time. The sample was small and the precision among prioritized edits was low, so it confirms that the system works technically, not that it would be effective in production.

**Limitations.** Only ten days of English Wikipedia, anonymous edits and articles. The label delay is assumed because the revert time is not available. `mw-reverted` does not distinguish vandalism from other reasons for reverting. The hyperparameter grid is small.

## How to run

The notebook is designed for **Google Colab**.

1. Click the Colab badge above and choose *Run all*.
2. The notebook installs `river` and downloads the data by itself.

A full run takes about 80 minutes, 30 of them for the live capture in section 12.1. For a quick test, lower `MUESTRA` (data loading cell) and `DEMO_SEG` (section 12.1). The libraries are listed in `requirements.txt`.
