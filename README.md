# Gym member segments

A gym sends every member the same monthly email, including a $10 extras credit. This analysis asks whether members fall into distinct groups, and whether sending that credit only to some groups is worth more than sending it to everyone.

The code refits the same 70/30 split (`random_state=42`) from `gym_members.csv`. It does not modify the CSV.

## Data

`gym_members.csv` has 330 members and 7 columns. Each row is one member, averaged over the last 3 months. There are no missing values.

`member_id` is an identifier and is left out of the model. The six clustering features are:

- `age`
- `visits_per_month`
- `avg_visit_minutes`
- `classes_per_month`
- `months_member`
- `monthly_extras_spend`

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python explore_data.py
python prepare_data.py
python elbow.py
python train_test_elbow.py
python cluster_summary.py
python credit_decision.py
```

`elbow.py` writes `elbow.png`. `train_test_elbow.py` writes `train_test_elbow.png`.

## Methodology

1. **Split.** `train_test_split` holds out 30% of members (99) and keeps 70% for training (231), with `random_state=42`.
2. **Scale.** `StandardScaler` is fit on the training features only, then applied to both the training and test features.
3. **Choose K.** K-means is fit on the scaled training data for K = 1 through 8 (`random_state=42`, `n_init=10`). Training inertia drops sharply through K = 4 (335.99) and then flattens (310.47 at K = 5). The elbow is at **K = 4**.
4. **Check the test members.** Each test member is assigned to the nearest training centroid. The model is not refit. Test inertia is 600.73 at K = 1 and 149.98 at K = 4, then falls only 8.59 from K = 4 to K = 5. The test curve has the same elbow.
5. **Name the segments.** All 330 members are assigned with the training-fitted model. Averages below are in the original units.

| Segment | Members | What stands out |
|---|---:|---|
| 0 Lightly engaged newcomers | 70 | Age 27.6, 3.1 visits, 6.3 months as a member, $9.30 extras |
| 1 Long-term members | 60 | Age 64.0, 58.7 months as a member, 12.0 visits, $16.40 extras |
| 2 Frequent independent exercisers | 109 | 16.0 visits and 64.4 minutes per visit, 0.9 classes, $24.60 extras |
| 3 Class-and-extras members | 91 | 6.9 classes a month and $46.90 extras |

## Credit decision

Finance's assumptions, applied to each member rather than to a segment average:

- A member who receives the credit uses it, which costs $10.
- A member who already spends $20 or more would spend the same without the credit, so the credit nets **−$10**.
- A member who spends less than $20 spends $25 more. At a 60% margin that is $15, so the credit nets **+$5**.
- A member who does not receive the credit nets $0.

Crediting a whole segment is worthwhile only when those member-level results add up to more than zero.

| Segment | Under $20 | $20 or more | Net if credited | Decision |
|---|---:|---:|---:|---|
| 0 Lightly engaged newcomers | 59 | 11 | 59×$5 + 11×−$10 = **+$185** | Send the credit |
| 1 Long-term members | 37 | 23 | 37×$5 + 23×−$10 = **−$45** | Do not send |
| 2 Frequent independent exercisers | 43 | 66 | 43×$5 + 66×−$10 = **−$445** | Do not send |
| 3 Class-and-extras members | 1 | 90 | 1×$5 + 90×−$10 = **−$895** | Do not send |

Cluster 1's average extras spend is under $20, but 23 members in that segment are at or above $20. Sending the credit to the whole segment still loses money, which is why the decision uses each member's spend.

`predict_member(30, 4, 42, 1, 5, 10)` places the example member in segment 0, Lightly engaged newcomers, so that member receives the credit.

**Baseline** (credit to all 330 members): 140×$5 + 190×−$10 = **−$1,200** a month.

**Targeted plan** (credit only to segment 0): 59×$5 + 11×−$10 = **+$185** a month.

Targeting improves the monthly result by **$1,385**.

## Files

| File | Role |
|---|---|
| `Clustering assignment.md` | Assignment brief |
| `gym_members.csv` | Member data |
| `explore_data.py` | Shape, preview, missing values, and the 70/30 split |
| `prepare_data.py` | Shared split and training-only scaler |
| `elbow.py` | Training elbow for K = 1 through 8 |
| `train_test_elbow.py` | Training inertia beside test inertia |
| `cluster_summary.py` | Segment sizes and original-unit averages |
| `credit_decision.py` | Segment choice, example member, and baseline versus targeted net |
| `elbow.png`, `train_test_elbow.png` | Saved plots |
