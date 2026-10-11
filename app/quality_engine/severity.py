def determine_severity(affected_rate):
    """
    Determine issue severity based on the
    proportion of affected data.

    Parameters:
        affected_rate (float): Affected proportion
        represented as a decimal between 0 and 1.

    Returns:
        str: Severity level.
    """

    if affected_rate <= 0:
        return "CLEAN"

    if affected_rate <= 0.05:
        return "LOW"

    if affected_rate <= 0.20:
        return "MEDIUM"

    if affected_rate <= 0.50:
        return "HIGH"

    return "CRITICAL"

if __name__ == "__main__":

    test_rates = [
        0,
        0.01,
        0.05,
        0.10,
        0.25,
        0.60
    ]

    for rate in test_rates:

        severity = determine_severity(rate)

        print(
            f"Rate: {rate * 100:.2f}% "
            f"→ Severity: {severity}"
        )