from finsight.data_generation.population import generate_population
from finsight.utils.random import create_rng


def main():

    rng = create_rng(42)

    population = generate_population(
        n_users=100,
        rng=rng
    )

    print("\nShape:")
    print(population.shape)

    print("\nFirst 5 users:")
    print(population.head())

    print("\nData types:")
    print(population.dtypes)

    print("\nMissing values:")
    print(population.isna().sum())

    print("\nCity distribution:")
    print(population["city_tier"].value_counts(normalize=True))

    print("\nEmployment distribution:")
    print(
        population["employment_type"]
        .value_counts(normalize=True)
    )

    print("\nNumerical summary:")
    print(
        population[
            [
                "age",
                "income_stability",
                "spending_propensity",
                "savings_propensity",
                "transaction_propensity",
                "obligation_burden"
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()