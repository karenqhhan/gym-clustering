"""Decide which K=4 segments get the $10 extras credit and value that choice.

Finance assumptions, applied to each member who receives the credit:
- The credit costs $10 and is always used.
- Extras spend of $20 or more does not change, so the credit nets -$10.
- Extras spend under $20 rises by $25. At a 60% margin that is $15, so the
  credit nets +$5.
A member who does not receive the credit contributes $0.
"""

import pandas as pd
from sklearn.cluster import KMeans

from prepare_data import FEATURE_COLUMNS, load_member_splits

LOW_SPEND_THRESHOLD = 20
LOW_SPEND_NET = 5
HIGH_SPEND_NET = -10
SEGMENT_NAMES = {
    0: "Lightly engaged newcomers",
    1: "Long-term members",
    2: "Frequent independent exercisers",
    3: "Class-and-extras members",
}
EXAMPLE_MEMBER = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}

_fitted = {}


def credit_net(extras_spend, receives_credit):
    """Monthly net from the credit for one member."""
    if not receives_credit:
        return 0
    if extras_spend < LOW_SPEND_THRESHOLD:
        return LOW_SPEND_NET
    return HIGH_SPEND_NET


def build_credit_model():
    """Fit K=4 on the scaled training members and choose profitable segments."""
    train_features, test_features, train_scaled, test_scaled, scaler = (
        load_member_splits()
    )
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    model.fit(train_scaled)

    labeled = pd.concat([train_features, test_features], axis=0)
    labels = pd.concat(
        [
            pd.Series(model.labels_, index=train_features.index),
            pd.Series(model.predict(test_scaled), index=test_features.index),
        ]
    )
    labeled = labeled.copy()
    labeled["cluster"] = labels
    labeled["net_if_credited"] = labeled["monthly_extras_spend"].map(
        lambda spend: credit_net(spend, True)
    )

    segment_net = labeled.groupby("cluster")["net_if_credited"].sum()
    credit_clusters = {
        int(cluster)
        for cluster, total in segment_net.items()
        if total > 0
    }

    _fitted["model"] = model
    _fitted["scaler"] = scaler
    _fitted["credit_clusters"] = credit_clusters
    return labeled, credit_clusters


def predict_member(
    age,
    visits_per_month,
    avg_visit_minutes,
    classes_per_month,
    months_member,
    monthly_extras_spend,
):
    """Return the segment and whether that segment receives the credit."""
    member = pd.DataFrame(
        [
            {
                "age": age,
                "visits_per_month": visits_per_month,
                "avg_visit_minutes": avg_visit_minutes,
                "classes_per_month": classes_per_month,
                "months_member": months_member,
                "monthly_extras_spend": monthly_extras_spend,
            }
        ]
    )
    scaled = pd.DataFrame(
        _fitted["scaler"].transform(member[FEATURE_COLUMNS]),
        columns=FEATURE_COLUMNS,
    )
    cluster = int(_fitted["model"].predict(scaled)[0])
    receives_credit = cluster in _fitted["credit_clusters"]
    return {
        "cluster": cluster,
        "segment": SEGMENT_NAMES[cluster],
        "receives_credit": receives_credit,
    }


def money(amount):
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}"


def plan_summary(members, receives_credit):
    """Apply the credit rule member by member and total the result."""
    nets = [
        credit_net(spend, credited)
        for spend, credited in zip(members["monthly_extras_spend"], receives_credit)
    ]
    low = sum(
        1
        for spend, credited in zip(members["monthly_extras_spend"], receives_credit)
        if credited and spend < LOW_SPEND_THRESHOLD
    )
    high = sum(
        1
        for spend, credited in zip(members["monthly_extras_spend"], receives_credit)
        if credited and spend >= LOW_SPEND_THRESHOLD
    )
    return {
        "low": low,
        "high": high,
        "low_net": low * LOW_SPEND_NET,
        "high_net": high * HIGH_SPEND_NET,
        "total": sum(nets),
    }


def main():
    members, credit_clusters = build_credit_model()

    print("Credit rule for one member who receives it:")
    print("  extras spend under $20: +$5")
    print("  extras spend of $20 or more: -$10")
    print("  no credit: $0")

    print("\nSegment economics if that segment receives the credit:")
    for cluster in range(4):
        segment = members[members["cluster"] == cluster]
        low = int((segment["monthly_extras_spend"] < LOW_SPEND_THRESHOLD).sum())
        high = int((segment["monthly_extras_spend"] >= LOW_SPEND_THRESHOLD).sum())
        total = low * LOW_SPEND_NET + high * HIGH_SPEND_NET
        decision = "credit" if cluster in credit_clusters else "no credit"
        print(
            f"  Cluster {cluster} ({SEGMENT_NAMES[cluster]}): "
            f"{len(segment)} members, {low} under $20, {high} at $20+"
        )
        print(
            f"    {low} x $5 + {high} x -$10 = {money(total)} -> {decision}"
        )

    chosen = ", ".join(
        f"{cluster} ({SEGMENT_NAMES[cluster]})" for cluster in sorted(credit_clusters)
    )
    print(f"\nSend the $10 credit to: {chosen}")

    example = predict_member(**EXAMPLE_MEMBER)
    print("\nExample member: age 30, 4 visits, 42 minutes, 1 class, 5 months, $10 extras")
    print(f"  Cluster {example['cluster']}: {example['segment']}")
    print(f"  Receives the credit: {example['receives_credit']}")

    baseline_flag = pd.Series(True, index=members.index)
    targeted_flag = members["cluster"].isin(credit_clusters)
    baseline = plan_summary(members, baseline_flag)
    targeted = plan_summary(members, targeted_flag)
    savings = targeted["total"] - baseline["total"]

    print("\nBaseline (all 330 members receive the credit):")
    print(
        f"  {baseline['low']} x $5 = {money(baseline['low_net'])}"
    )
    print(
        f"  {baseline['high']} x -$10 = {money(baseline['high_net'])}"
    )
    print(f"  Monthly net = {money(baseline['total'])}")

    print("\nTargeted plan (credit only to the selected segments):")
    print(
        f"  {targeted['low']} x $5 = {money(targeted['low_net'])}"
    )
    print(
        f"  {targeted['high']} x -$10 = {money(targeted['high_net'])}"
    )
    print(f"  Monthly net = {money(targeted['total'])}")
    print(f"\nMonthly improvement over the baseline: {money(savings)}")


if __name__ == "__main__":
    main()
