import pandas as pd

from finsight.data_generation.population import generate_population
from finsight.data_generation.income import (
    generate_base_income,
    generate_salary_days,
    generate_monthly_income,
)
from finsight.utils.random import create_rng


def main():

    rng = create_rng(42)

    # --------------------------------------------------------
    # Generate population
    # --------------------------------------------------------

    population = generate_population(
        n_users=100,
        rng=rng
    )

    # --------------------------------------------------------
    # Generate base income
    # --------------------------------------------------------

    population = generate_base_income(
        population,
        rng
    )

    # --------------------------------------------------------
    # Generate salary days
    # --------------------------------------------------------

    population = generate_salary_days(
        population,
        rng
    )

    # --------------------------------------------------------
    # Generate monthly income
    # --------------------------------------------------------

    monthly_income = generate_monthly_income(
        population,
        n_months=12,
        rng=rng
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nPopulation shape:")
    print(population.shape)

    print("\nMonthly income shape:")
    print(monthly_income.shape)

    print("\nPopulation columns:")
    print(population.columns.tolist())

    print("\nMonthly income columns:")
    print(monthly_income.columns.tolist())

    print("\nMissing values:")
    print(monthly_income.isna().sum())

    print("\nBase income summary:")
    print(
        population["base_income"].describe()
    )

    print("\nMonthly salary summary:")
    print(
        monthly_income["monthly_salary"].describe()
    )

    print("\nTotal income summary:")
    print(
        monthly_income["total_income"].describe()
    )

    print("\nSalary-day distribution:")
    print(
        population["salary_day"]
        .value_counts()
        .sort_index()
    )

    print("\nSample monthly income:")
    print(
        monthly_income.head(15)
    )

    # --------------------------------------------------------
    # User-month count validation
    # --------------------------------------------------------

    expected_rows = 100 * 12

    assert len(monthly_income) == expected_rows

    print(
        f"\nExpected rows: {expected_rows}"
    )

    print(
        f"Actual rows:   {len(monthly_income)}"
    )

    print(
        "\nIncome generation validation passed."
    )


if __name__ == "__main__":
    main()