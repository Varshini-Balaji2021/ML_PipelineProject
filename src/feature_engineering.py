"""
feature_engineering_plan.py

PURPOSE
-------
Create a formal Feature Engineering Plan from the EDA analytical outputs.

This stage does not create or transform any model features. It defines:

- which features are proposed;
- why each feature is proposed;
- which source variables are required;
- whether the feature is automatic or optional;
- which analytical branches may use it;
- what validation is required before adoption;
- which script should eventually implement it.

EXPECTED PIPELINE POSITION
--------------------------
    data_understanding.py
        ↓
    eda.py
        ↓
    eda_analysis_summary.py
        ↓
    feature_engineering_plan.py
        ↓
    feature engineering execution
        ↓
    model-specific feature selection
        ↓
    model development

FEATURE GROUPS
--------------
1. Deterministic features
   Mathematically defined ratios and transformations with direct meaning.

2. Domain features
   Features proposed from business or subject knowledge.

3. Statistical features
   Transformations motivated by EDA, curvature, skewness, or interactions.

4. Machine-learning features
   Predictive representations whose value is assessed through validation.

5. Inferential features
   Interpretable or estimable representations for statistical modelling.

INPUTS
------
Training split:

    data/splits/train.csv

EDA analytical outputs:

    results/eda_analysis_summary/
        candidate_explanatory_variables.csv
        predictor_correlation_screen.csv
        nonlinearity_screen.csv
        feature_engineering_recommendations.csv
        modelling_challenges.csv
        business_interpretation.csv
        model_development_recommendations.csv

OUTPUTS
-------
    results/feature_engineering_plan/
        feature_engineering_plan.csv
        feature_repository.csv
        deterministic_feature_plan.csv
        domain_feature_plan.csv
        statistical_feature_plan.csv
        ml_feature_plan.csv
        inference_feature_plan.csv
        source_variable_dependencies.csv
        feature_engineering_plan.md
        feature_engineering_plan.html
        feature_engineering_plan_summary.txt

RUN
---
From the project root:

    python src/feature_engineering_plan.py
"""

from __future__ import annotations

from pathlib import Path
from sys import path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# 1. Project paths and semantic schema
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
path.insert(0, str(PROJECT_ROOT))

from src.semantic_schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    TARGET_COLUMN,
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

EDA_SUMMARY_DIR = (
    PROJECT_ROOT
    / "results"
    / "eda_analysis_summary"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "feature_engineering_plan"
)

MASTER_PLAN_PATH = (
    OUTPUT_DIR
    / "feature_engineering_plan.csv"
)

FEATURE_REPOSITORY_PATH = (
    OUTPUT_DIR
    / "feature_repository.csv"
)

DETERMINISTIC_PLAN_PATH = (
    OUTPUT_DIR
    / "deterministic_feature_plan.csv"
)

DOMAIN_PLAN_PATH = (
    OUTPUT_DIR
    / "domain_feature_plan.csv"
)

STATISTICAL_PLAN_PATH = (
    OUTPUT_DIR
    / "statistical_feature_plan.csv"
)

ML_PLAN_PATH = (
    OUTPUT_DIR
    / "ml_feature_plan.csv"
)

INFERENCE_PLAN_PATH = (
    OUTPUT_DIR
    / "inference_feature_plan.csv"
)

DEPENDENCIES_PATH = (
    OUTPUT_DIR
    / "source_variable_dependencies.csv"
)

MARKDOWN_PATH = (
    OUTPUT_DIR
    / "feature_engineering_plan.md"
)

HTML_PATH = (
    OUTPUT_DIR
    / "feature_engineering_plan.html"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "feature_engineering_plan_summary.txt"
)


# =========================================================
# 2. Input helpers
# =========================================================

def require_training_data() -> None:
    """Ensure the training split exists."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}\n"
            "Run split.py first."
        )


def read_optional_csv(
    filename: str,
) -> pd.DataFrame | None:
    """Read an EDA analytical output when available."""

    file_path = (
        EDA_SUMMARY_DIR
        / filename
    )

    if not file_path.exists():
        return None

    data = pd.read_csv(
        file_path
    )

    return (
        None
        if data.empty
        else data
    )


def source_columns_exist(
    data: pd.DataFrame,
    source_columns: list[str],
) -> bool:
    """Check whether all source variables exist."""

    return set(
        source_columns
    ).issubset(
        data.columns
    )


def safe_division_possible(
    data: pd.DataFrame,
    denominator: str,
) -> tuple[bool, int]:
    """Check zero denominators before proposing a ratio."""

    if denominator not in data.columns:
        return False, 0

    numeric = pd.to_numeric(
        data[denominator],
        errors="coerce",
    )

    zero_count = int(
        (
            numeric.notna()
            & np.isclose(
                numeric,
                0,
            )
        ).sum()
    )

    return zero_count == 0, zero_count


# =========================================================
# 3. Common row factory
# =========================================================

def plan_row(
    *,
    feature_name: str,
    feature_group: str,
    feature_subtype: str,
    source_columns: list[str],
    formula_or_method: str,
    business_meaning: str,
    analytical_rationale: str,
    creation_status: str,
    used_by_ml: bool,
    used_by_inference: bool,
    used_by_optimization: bool = False,
    used_by_grey_models: bool = False,
    requires_training_fit: bool = False,
    validation_rule: str = "",
    leakage_risk: str = "LOW",
    implementation_module: str = "",
    priority: str = "MEDIUM",
    analyst_decision: str = "PROPOSED",
    notes: str = "",
) -> dict[str, Any]:
    """Create one standardized feature-plan row."""

    return {
        "feature_name": feature_name,
        "feature_group": feature_group,
        "feature_subtype": feature_subtype,
        "source_columns": " | ".join(
            source_columns
        ),
        "formula_or_method": (
            formula_or_method
        ),
        "business_meaning": (
            business_meaning
        ),
        "analytical_rationale": (
            analytical_rationale
        ),
        "creation_status": (
            creation_status
        ),
        "used_by_machine_learning": (
            used_by_ml
        ),
        "used_by_inference": (
            used_by_inference
        ),
        "used_by_optimization": (
            used_by_optimization
        ),
        "used_by_grey_models": (
            used_by_grey_models
        ),
        "requires_training_fit": (
            requires_training_fit
        ),
        "validation_rule": (
            validation_rule
        ),
        "leakage_risk": (
            leakage_risk
        ),
        "implementation_module": (
            implementation_module
        ),
        "priority": priority,
        "analyst_decision": (
            analyst_decision
        ),
        "notes": notes,
    }


# =========================================================
# 4. Deterministic feature plan
# =========================================================

def deterministic_feature_candidates(
    data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create mathematically defined ratio proposals."""

    definitions = [
        {
            "feature_name": (
                "rooms_per_household"
            ),
            "numerator": (
                "total_rooms"
            ),
            "denominator": (
                "households"
            ),
            "business_meaning": (
                "Average housing-space availability per household."
            ),
        },
        {
            "feature_name": (
                "bedrooms_per_room"
            ),
            "numerator": (
                "total_bedrooms"
            ),
            "denominator": (
                "total_rooms"
            ),
            "business_meaning": (
                "Bedroom share within the housing stock."
            ),
        },
        {
            "feature_name": (
                "population_per_household"
            ),
            "numerator": (
                "population"
            ),
            "denominator": (
                "households"
            ),
            "business_meaning": (
                "Average household occupancy."
            ),
        },
        {
            "feature_name": (
                "bedrooms_per_household"
            ),
            "numerator": (
                "total_bedrooms"
            ),
            "denominator": (
                "households"
            ),
            "business_meaning": (
                "Average bedroom availability per household."
            ),
        },
        {
            "feature_name": (
                "rooms_per_person"
            ),
            "numerator": (
                "total_rooms"
            ),
            "denominator": (
                "population"
            ),
            "business_meaning": (
                "Housing-space availability per person."
            ),
        },
        {
            "feature_name": (
                "bedrooms_per_person"
            ),
            "numerator": (
                "total_bedrooms"
            ),
            "denominator": (
                "population"
            ),
            "business_meaning": (
                "Bedroom availability per person."
            ),
        },
    ]

    rows = []

    for definition in definitions:
        numerator = (
            definition[
                "numerator"
            ]
        )

        denominator = (
            definition[
                "denominator"
            ]
        )

        required = [
            numerator,
            denominator,
        ]

        if not source_columns_exist(
            data,
            required,
        ):
            continue

        safe, zero_count = (
            safe_division_possible(
                data,
                denominator,
            )
        )

        validation_rule = (
            "Denominator must be non-zero; preserve missing values."
        )

        notes = (
            "No zero denominators detected in training data."
            if safe
            else (
                f"{zero_count} zero denominator value(s) detected; "
                "define a safe-division policy before execution."
            )
        )

        rows.append(
            plan_row(
                feature_name=(
                    definition[
                        "feature_name"
                    ]
                ),
                feature_group=(
                    "Deterministic"
                ),
                feature_subtype=(
                    "Ratio"
                ),
                source_columns=required,
                formula_or_method=(
                    f"{numerator} / {denominator}"
                ),
                business_meaning=(
                    definition[
                        "business_meaning"
                    ]
                ),
                analytical_rationale=(
                    "Separates observational-unit scale from "
                    "density, composition, or resource availability."
                ),
                creation_status=(
                    "AUTOMATIC CANDIDATE"
                ),
                used_by_ml=True,
                used_by_inference=True,
                used_by_optimization=True,
                used_by_grey_models=True,
                requires_training_fit=False,
                validation_rule=(
                    validation_rule
                ),
                leakage_risk="LOW",
                implementation_module=(
                    "deterministic_features.py"
                ),
                priority="HIGH",
                analyst_decision=(
                    "APPROVE FOR IMPLEMENTATION"
                    if safe
                    else "REVIEW ZERO-DENOMINATOR POLICY"
                ),
                notes=notes,
            )
        )

    return rows


# =========================================================
# 5. Domain feature plan
# =========================================================

def domain_feature_candidates(
    data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Create domain-informed proposals.

    These are not automatically created unless they can be constructed
    from currently available data.
    """

    rows = []

    if "ocean_proximity" in data.columns:
        rows.append(
            plan_row(
                feature_name=(
                    "coastal_access_group"
                ),
                feature_group="Domain",
                feature_subtype=(
                    "Category consolidation"
                ),
                source_columns=[
                    "ocean_proximity",
                ],
                formula_or_method=(
                    "Business-approved grouping of proximity categories"
                ),
                business_meaning=(
                    "Simplified market-access or coastal-amenity segment."
                ),
                analytical_rationale=(
                    "May improve interpretability when individual categories "
                    "are sparse or operational decisions use broader segments."
                ),
                creation_status=(
                    "OPTIONAL DOMAIN CANDIDATE"
                ),
                used_by_ml=True,
                used_by_inference=True,
                requires_training_fit=False,
                validation_rule=(
                    "Grouping must be approved before use and applied "
                    "identically to every dataset split."
                ),
                leakage_risk="LOW",
                implementation_module=(
                    "domain_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "REQUIRES DOMAIN APPROVAL"
                ),
                notes=(
                    "Do not collapse categories only to improve model fit."
                ),
            )
        )

    rows.extend(
        [
            plan_row(
                feature_name=(
                    "distance_to_business_centre"
                ),
                feature_group="Domain",
                feature_subtype=(
                    "External geographic feature"
                ),
                source_columns=[
                    "longitude",
                    "latitude",
                    "external_cbd_coordinates",
                ],
                formula_or_method=(
                    "Geodesic distance to a defined central business centre"
                ),
                business_meaning=(
                    "Accessibility to major employment and commercial centres."
                ),
                analytical_rationale=(
                    "Coordinates alone may not express accessibility clearly."
                ),
                creation_status=(
                    "EXTERNAL DATA REQUIRED"
                ),
                used_by_ml=True,
                used_by_inference=True,
                used_by_optimization=True,
                requires_training_fit=False,
                validation_rule=(
                    "External reference location and distance method "
                    "must be documented."
                ),
                leakage_risk="LOW",
                implementation_module=(
                    "domain_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "DEFER UNTIL EXTERNAL DATA AVAILABLE"
                ),
            ),
            plan_row(
                feature_name=(
                    "local_amenity_index"
                ),
                feature_group="Domain",
                feature_subtype=(
                    "External composite index"
                ),
                source_columns=[
                    "external_amenity_data",
                ],
                formula_or_method=(
                    "Business-approved composite score"
                ),
                business_meaning=(
                    "Local access to schools, services, transport, "
                    "and amenities."
                ),
                analytical_rationale=(
                    "Addresses possible omitted neighbourhood characteristics."
                ),
                creation_status=(
                    "EXTERNAL DATA REQUIRED"
                ),
                used_by_ml=True,
                used_by_inference=True,
                used_by_optimization=True,
                requires_training_fit=False,
                validation_rule=(
                    "Index construction and weighting must be documented."
                ),
                leakage_risk="MEDIUM",
                implementation_module=(
                    "domain_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "DEFER UNTIL EXTERNAL DATA AVAILABLE"
                ),
            ),
        ]
    )

    return rows


# =========================================================
# 6. Statistical feature plan
# =========================================================

def statistical_feature_candidates(
    data: pd.DataFrame,
    nonlinearity: pd.DataFrame | None,
    modelling_challenges: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Create EDA-motivated statistical feature proposals."""

    rows = []

    nonlinear_features: list[str] = []

    if (
        nonlinearity is not None
        and "feature"
        in nonlinearity.columns
    ):
        if "exploratory_pattern" in nonlinearity.columns:
            nonlinear_features = (
                nonlinearity.loc[
                    nonlinearity[
                        "exploratory_pattern"
                    ].isin(
                        [
                            "Moderate curvature",
                            "Strong nonlinear pattern",
                        ]
                    ),
                    "feature",
                ]
                .tolist()
            )
        else:
            nonlinear_features = (
                nonlinearity[
                    "feature"
                ]
                .tolist()
            )

    for feature in nonlinear_features:
        if feature not in data.columns:
            continue

        rows.append(
            plan_row(
                feature_name=(
                    f"{feature}_squared"
                ),
                feature_group=(
                    "Statistical"
                ),
                feature_subtype=(
                    "Polynomial term"
                ),
                source_columns=[
                    feature,
                ],
                formula_or_method=(
                    f"{feature} ** 2"
                ),
                business_meaning=(
                    f"Allows the marginal relationship of {feature} "
                    "to change with its level."
                ),
                analytical_rationale=(
                    "Proposed because the EDA LOWESS screen suggests "
                    "possible curvature."
                ),
                creation_status=(
                    "MODEL-SPECIFICATION CANDIDATE"
                ),
                used_by_ml=False,
                used_by_inference=True,
                requires_training_fit=False,
                validation_rule=(
                    "Retain only when specification, diagnostics, "
                    "and validation support the nonlinear term."
                ),
                leakage_risk="LOW",
                implementation_module=(
                    "statistical_features.py"
                ),
                priority=(
                    "HIGH"
                    if feature
                    in {
                        "longitude",
                        "latitude",
                    }
                    else "MEDIUM"
                ),
                analyst_decision=(
                    "EVALUATE, DO NOT IMPOSE"
                ),
            )
        )

    interaction_definitions = [
        {
            "feature_name": (
                "longitude_latitude_interaction"
            ),
            "source_columns": [
                "longitude",
                "latitude",
            ],
            "formula": (
                "longitude * latitude"
            ),
            "meaning": (
                "Joint geographic effect."
            ),
        },
        {
            "feature_name": (
                "income_ocean_proximity_interaction"
            ),
            "source_columns": [
                "median_income",
                "ocean_proximity",
            ],
            "formula": (
                "median_income × one-hot ocean-proximity indicators"
            ),
            "meaning": (
                "Allows the income relationship to differ across "
                "location categories."
            ),
        },
        {
            "feature_name": (
                "income_age_interaction"
            ),
            "source_columns": [
                "median_income",
                "housing_median_age",
            ],
            "formula": (
                "median_income * housing_median_age"
            ),
            "meaning": (
                "Tests whether the relationship between income and "
                "the response varies with housing age."
            ),
        },
    ]

    for definition in interaction_definitions:
        if not source_columns_exist(
            data,
            definition[
                "source_columns"
            ],
        ):
            continue

        rows.append(
            plan_row(
                feature_name=(
                    definition[
                        "feature_name"
                    ]
                ),
                feature_group=(
                    "Statistical"
                ),
                feature_subtype=(
                    "Interaction"
                ),
                source_columns=(
                    definition[
                        "source_columns"
                    ]
                ),
                formula_or_method=(
                    definition[
                        "formula"
                    ]
                ),
                business_meaning=(
                    definition[
                        "meaning"
                    ]
                ),
                analytical_rationale=(
                    "Proposed as an interaction hypothesis; not justified "
                    "by marginal effects alone."
                ),
                creation_status=(
                    "HYPOTHESIS CANDIDATE"
                ),
                used_by_ml=True,
                used_by_inference=True,
                requires_training_fit=False,
                validation_rule=(
                    "Retain only with conceptual justification and "
                    "incremental empirical value."
                ),
                leakage_risk="LOW",
                implementation_module=(
                    "statistical_features.py"
                ),
                priority="MEDIUM",
                analyst_decision=(
                    "EVALUATE, DO NOT IMPOSE"
                ),
            )
        )

    return rows


# =========================================================
# 7. Machine-learning feature plan
# =========================================================

def ml_feature_candidates(
    data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create predictive feature-representation proposals."""

    rows = []

    rows.append(
        plan_row(
            feature_name=(
                "standardized_numeric_predictors"
            ),
            feature_group=(
                "Machine Learning"
            ),
            feature_subtype=(
                "Scaling"
            ),
            source_columns=[
                column
                for column in (
                    NUMERICAL_COLUMNS
                )
                if column
                in data.columns
            ],
            formula_or_method=(
                "StandardScaler fitted on training data"
            ),
            business_meaning=(
                "Places numerical predictors on comparable scales."
            ),
            analytical_rationale=(
                "Required for stable gradient-based optimisation."
            ),
            creation_status=(
                "MANDATORY FOR SGD PIPELINE"
            ),
            used_by_ml=True,
            used_by_inference=False,
            requires_training_fit=True,
            validation_rule=(
                "Fit only on training data; reuse unchanged for "
                "validation, test, and future observations."
            ),
            leakage_risk="HIGH",
            implementation_module=(
                "ml_features.py"
            ),
            priority="HIGH",
            analyst_decision=(
                "APPROVE FOR ML"
            ),
        )
    )

    if CATEGORICAL_COLUMNS:
        rows.append(
            plan_row(
                feature_name=(
                    "one_hot_categorical_predictors"
                ),
                feature_group=(
                    "Machine Learning"
                ),
                feature_subtype=(
                    "Categorical encoding"
                ),
                source_columns=[
                    column
                    for column in (
                        CATEGORICAL_COLUMNS
                    )
                    if column
                    in data.columns
                ],
                formula_or_method=(
                    "OneHotEncoder(handle_unknown='ignore') "
                    "fitted on training data"
                ),
                business_meaning=(
                    "Represents categorical differences without "
                    "imposing numerical ordering."
                ),
                analytical_rationale=(
                    "Required by linear and gradient-based estimators."
                ),
                creation_status=(
                    "MANDATORY FOR ML PIPELINE"
                ),
                used_by_ml=True,
                used_by_inference=False,
                requires_training_fit=True,
                validation_rule=(
                    "Fit category levels using training data only."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "ml_features.py"
                ),
                priority="HIGH",
                analyst_decision=(
                    "APPROVE FOR ML"
                ),
            )
        )

    rows.extend(
        [
            plan_row(
                feature_name=(
                    "polynomial_feature_expansion"
                ),
                feature_group=(
                    "Machine Learning"
                ),
                feature_subtype=(
                    "Polynomial representation"
                ),
                source_columns=[
                    column
                    for column in (
                        NUMERICAL_COLUMNS
                    )
                    if column
                    in data.columns
                ],
                formula_or_method=(
                    "PolynomialFeatures degree selected through validation"
                ),
                business_meaning=(
                    "Flexible predictive representation of curvature "
                    "and interactions."
                ),
                analytical_rationale=(
                    "May improve prediction when linear SGD underfits."
                ),
                creation_status=(
                    "OPTIONAL VALIDATION CANDIDATE"
                ),
                used_by_ml=True,
                used_by_inference=False,
                requires_training_fit=True,
                validation_rule=(
                    "Select complexity using validation data; "
                    "never choose degree using test performance."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "ml_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "DEFER UNTIL BASELINE VALIDATION"
                ),
            ),
            plan_row(
                feature_name=(
                    "spline_basis_for_prediction"
                ),
                feature_group=(
                    "Machine Learning"
                ),
                feature_subtype=(
                    "Spline representation"
                ),
                source_columns=[
                    column
                    for column in (
                        NUMERICAL_COLUMNS
                    )
                    if column
                    in data.columns
                ],
                formula_or_method=(
                    "SplineTransformer fitted on training data"
                ),
                business_meaning=(
                    "Flexible smooth predictive relationships."
                ),
                analytical_rationale=(
                    "Alternative to high-degree polynomial expansion."
                ),
                creation_status=(
                    "OPTIONAL VALIDATION CANDIDATE"
                ),
                used_by_ml=True,
                used_by_inference=False,
                requires_training_fit=True,
                validation_rule=(
                    "Select knots and degree through validation."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "ml_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "DEFER UNTIL BASELINE VALIDATION"
                ),
            ),
        ]
    )

    return rows


# =========================================================
# 8. Inferential feature plan
# =========================================================

def inference_feature_candidates(
    data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create interpretable and estimable feature proposals."""

    rows = []

    rows.extend(
        [
            plan_row(
                feature_name=(
                    "centered_numeric_predictors"
                ),
                feature_group=(
                    "Inferential"
                ),
                feature_subtype=(
                    "Centering"
                ),
                source_columns=[
                    column
                    for column in (
                        NUMERICAL_COLUMNS
                    )
                    if column
                    in data.columns
                ],
                formula_or_method=(
                    "x - training mean"
                ),
                business_meaning=(
                    "Makes the intercept and interaction terms more "
                    "interpretable."
                ),
                analytical_rationale=(
                    "Reduces nonessential collinearity in polynomial "
                    "and interaction specifications."
                ),
                creation_status=(
                    "OPTIONAL INFERENTIAL CANDIDATE"
                ),
                used_by_ml=False,
                used_by_inference=True,
                requires_training_fit=True,
                validation_rule=(
                    "Estimate centering constants from training data "
                    "and preserve them for validation and test."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "inference_features.py"
                ),
                priority="MEDIUM",
                analyst_decision=(
                    "USE WHEN POLYNOMIAL OR INTERACTION TERMS ENTER"
                ),
            ),
            plan_row(
                feature_name=(
                    "orthogonal_polynomial_terms"
                ),
                feature_group=(
                    "Inferential"
                ),
                feature_subtype=(
                    "Orthogonal polynomial"
                ),
                source_columns=[
                    column
                    for column in (
                        NUMERICAL_COLUMNS
                    )
                    if column
                    in data.columns
                ],
                formula_or_method=(
                    "Orthogonal polynomial basis for selected variables"
                ),
                business_meaning=(
                    "Represents curvature while reducing collinearity "
                    "among polynomial terms."
                ),
                analytical_rationale=(
                    "Useful when ordinary squared terms create unstable "
                    "coefficient estimates."
                ),
                creation_status=(
                    "MODEL-SPECIFICATION CANDIDATE"
                ),
                used_by_ml=False,
                used_by_inference=True,
                requires_training_fit=True,
                validation_rule=(
                    "Use only for variables supported by specification "
                    "and diagnostics."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "inference_features.py"
                ),
                priority="LOW",
                analyst_decision=(
                    "DEFER UNTIL OLS SPECIFICATION TEST"
                ),
            ),
            plan_row(
                feature_name=(
                    "interpretable_spline_basis"
                ),
                feature_group=(
                    "Inferential"
                ),
                feature_subtype=(
                    "Spline basis"
                ),
                source_columns=[
                    "longitude",
                    "latitude",
                    "median_income",
                ],
                formula_or_method=(
                    "Restricted spline basis with documented knots"
                ),
                business_meaning=(
                    "Captures smooth nonlinear relationships while "
                    "preserving interpretable effect plots."
                ),
                analytical_rationale=(
                    "Candidate when linear and polynomial specifications "
                    "remain inadequate."
                ),
                creation_status=(
                    "GAM / SPLINE CANDIDATE"
                ),
                used_by_ml=False,
                used_by_inference=True,
                requires_training_fit=True,
                validation_rule=(
                    "Choose smoothness and knots without using the test set."
                ),
                leakage_risk="HIGH",
                implementation_module=(
                    "inference_features.py"
                ),
                priority="MEDIUM",
                analyst_decision=(
                    "DEFER UNTIL LINEAR AND POLYNOMIAL ASSESSMENT"
                ),
            ),
        ]
    )

    return rows


# =========================================================
# 9. Repository and dependency outputs
# =========================================================

def create_feature_repository(
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise feature-repository view."""

    repository = plan[
        [
            "feature_name",
            "feature_group",
            "feature_subtype",
            "source_columns",
            "business_meaning",
            "creation_status",
            "used_by_machine_learning",
            "used_by_inference",
            "used_by_optimization",
            "used_by_grey_models",
            "requires_training_fit",
            "implementation_module",
            "priority",
            "analyst_decision",
        ]
    ].copy()

    repository.insert(
        0,
        "feature_id",
        [
            f"FEAT-{index:03d}"
            for index in range(
                1,
                len(repository) + 1,
            )
        ],
    )

    repository[
        "repository_status"
    ] = "PLANNED"

    return repository


def create_dependency_table(
    plan: pd.DataFrame,
) -> pd.DataFrame:
    """Expand source-column dependencies into long format."""

    rows = []

    for _, feature in plan.iterrows():
        source_text = str(
            feature[
                "source_columns"
            ]
        )

        sources = [
            item.strip()
            for item in source_text.split("|")
            if item.strip()
        ]

        for source in sources:
            rows.append(
                {
                    "feature_name": (
                        feature[
                            "feature_name"
                        ]
                    ),
                    "feature_group": (
                        feature[
                            "feature_group"
                        ]
                    ),
                    "source_variable": (
                        source
                    ),
                    "source_available_in_training_data": (
                        source
                        in pd.read_csv(
                            TRAIN_PATH,
                            nrows=0,
                        ).columns
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# 10. Reporting helpers
# =========================================================

def dataframe_markdown(
    data: pd.DataFrame,
) -> str:
    """Create a simple Markdown table."""

    if data.empty:
        return "_No proposed features._"

    columns = list(
        data.columns
    )

    lines = [
        "| "
        + " | ".join(
            columns
        )
        + " |",
        "| "
        + " | ".join(
            ["---"] * len(
                columns
            )
        )
        + " |",
    ]

    for _, row in data.iterrows():
        values = []

        for column in columns:
            value = row[
                column
            ]

            if pd.isna(value):
                text = ""
            elif isinstance(
                value,
                float,
            ):
                text = f"{value:.4f}"
            else:
                text = str(
                    value
                )

            values.append(
                text.replace(
                    "|",
                    "/",
                )
            )

        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def create_markdown_report(
    plan: pd.DataFrame,
    repository: pd.DataFrame,
) -> str:
    """Create the feature-engineering plan report."""

    group_sections = []

    for group, group_data in plan.groupby(
        "feature_group",
        sort=False,
    ):
        group_sections.append(
            f"""## {group} Features

{dataframe_markdown(group_data)}
"""
        )

    return f"""# Feature Engineering Plan

## Purpose

This plan converts EDA findings into a governed set of proposed features.
It does not create the variables. Execution should occur only after the
analyst approves the relevant proposals.

## Planning Principles

1. Deterministic ratios may be created automatically when their source
   columns and denominator rules are valid.
2. Domain features require subject-matter approval.
3. Statistical features remain hypotheses until specification and
   diagnostics support them.
4. Machine-learning transformations must be fitted on training data only.
5. Inferential features must preserve interpretation and estimability.
6. The test set must never determine feature creation or selection.

## Feature Repository

{dataframe_markdown(repository)}

{chr(10).join(group_sections)}
"""


def html_table(
    data: pd.DataFrame,
) -> str:
    """Create a styled HTML table."""

    if data.empty:
        return "<p><em>No proposed features.</em></p>"

    return data.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
    )


def create_html_report(
    plan: pd.DataFrame,
    repository: pd.DataFrame,
) -> str:
    """Create the HTML Feature Engineering Plan."""

    cards = []

    for group, count in (
        plan[
            "feature_group"
        ]
        .value_counts()
        .items()
    ):
        cards.append(
            f"""
            <div class="summary-card">
                <div class="card-number">{int(count)}</div>
                <div class="card-label">{group}</div>
            </div>
            """
        )

    group_sections = []

    for group, group_data in plan.groupby(
        "feature_group",
        sort=False,
    ):
        group_sections.append(
            f"""
            <section class="section" id="{group.lower().replace(' ', '-')}">
                <h2>{group} Features</h2>
                <div class="table-wrap">
                    {html_table(group_data)}
                </div>
            </section>
            """
        )

    css = """
    :root {
        --navy: #16324f;
        --teal: #2b7a78;
        --blue: #2d6a8a;
        --gold: #c6922e;
        --green: #4f772d;
        --light: #f4f7f9;
        --border: #d8e1e7;
        --text: #1f2933;
    }

    * { box-sizing: border-box; }

    body {
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        background: #eaf0f4;
        color: var(--text);
        line-height: 1.55;
    }

    .page {
        max-width: 1500px;
        margin: auto;
        padding: 30px;
    }

    .hero {
        padding: 32px;
        color: white;
        border-radius: 17px;
        background:
            linear-gradient(
                135deg,
                var(--navy),
                var(--teal)
            );
    }

    .hero h1 {
        margin-top: 0;
    }

    .summary-grid {
        display: grid;
        grid-template-columns:
            repeat(
                auto-fit,
                minmax(160px, 1fr)
            );
        gap: 12px;
        margin-top: 20px;
    }

    .summary-card {
        padding: 16px;
        color: white;
        border-radius: 11px;
        background: var(--blue);
    }

    .card-number {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .section {
        margin-top: 22px;
        padding: 24px;
        background: white;
        border-radius: 14px;
        box-shadow:
            0 4px 14px
            rgba(0,0,0,.06);
    }

    .section h2 {
        margin-top: 0;
        color: var(--navy);
        border-bottom:
            3px solid
            var(--teal);
        padding-bottom: 8px;
    }

    .principles {
        background: #fff7db;
        border-left:
            5px solid
            var(--gold);
        padding: 16px;
        border-radius: 9px;
    }

    .workflow {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }

    .step {
        padding: 10px 13px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--light);
        font-weight: 600;
    }

    .arrow {
        color: var(--teal);
        font-size: 1.25rem;
    }

    .table-wrap {
        overflow-x: auto;
    }

    table.data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: .82rem;
    }

    .data-table th {
        padding: 8px;
        color: white;
        background: var(--navy);
        text-align: left;
        white-space: nowrap;
    }

    .data-table td {
        padding: 8px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
    }

    .data-table tr:nth-child(even) {
        background: #f8fafb;
    }

    @media print {
        body {
            background: white;
        }

        .section {
            box-shadow: none;
        }
    }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>
<title>Feature Engineering Plan</title>
<style>{css}</style>
</head>
<body>
<div class="page">

<section class="hero">
<h1>Feature Engineering Plan</h1>
<p>
This document converts EDA findings into a governed repository of
proposed deterministic, domain, statistical, machine-learning, and
inferential features.
</p>

<div class="summary-grid">
{''.join(cards)}
</div>
</section>

<section class="section">
<h2>Feature Engineering Governance</h2>
<div class="principles">
<ol>
<li>EDA proposes features; it does not automatically approve all of them.</li>
<li>Deterministic ratios may be executed when mathematical rules are valid.</li>
<li>Domain features require subject-matter approval.</li>
<li>Statistical and interaction features remain hypotheses.</li>
<li>ML transformations must be fitted on training data only.</li>
<li>Inferential features must satisfy interpretation and estimability requirements.</li>
<li>The test set must never determine feature creation or selection.</li>
</ol>
</div>
</section>

<section class="section">
<h2>Planned Workflow</h2>
<div class="workflow">
<div class="step">EDA Findings</div>
<div class="arrow">→</div>
<div class="step">Feature Engineering Plan</div>
<div class="arrow">→</div>
<div class="step">Analyst Approval</div>
<div class="arrow">→</div>
<div class="step">Feature Repository</div>
<div class="arrow">→</div>
<div class="step">Model-Specific Selection</div>
<div class="arrow">→</div>
<div class="step">Model Development</div>
</div>
</section>

<section class="section">
<h2>Feature Repository</h2>
<div class="table-wrap">
{html_table(repository)}
</div>
</section>

{''.join(group_sections)}

</div>
</body>
</html>
"""


# =========================================================
# 11. Main
# =========================================================

def main() -> None:
    """Generate the complete Feature Engineering Plan."""

    require_training_data()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        TRAIN_PATH
    )

    nonlinearity = read_optional_csv(
        "nonlinearity_screen.csv"
    )

    modelling_challenges = read_optional_csv(
        "modelling_challenges.csv"
    )

    rows: list[
        dict[str, Any]
    ] = []

    rows.extend(
        deterministic_feature_candidates(
            data
        )
    )

    rows.extend(
        domain_feature_candidates(
            data
        )
    )

    rows.extend(
        statistical_feature_candidates(
            data,
            nonlinearity,
            modelling_challenges,
        )
    )

    rows.extend(
        ml_feature_candidates(
            data
        )
    )

    rows.extend(
        inference_feature_candidates(
            data
        )
    )

    plan = pd.DataFrame(
        rows
    )

    group_order = [
        "Deterministic",
        "Domain",
        "Statistical",
        "Machine Learning",
        "Inferential",
    ]

    plan[
        "feature_group"
    ] = pd.Categorical(
        plan[
            "feature_group"
        ],
        categories=group_order,
        ordered=True,
    )

    plan = (
        plan.sort_values(
            [
                "feature_group",
                "priority",
                "feature_name",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    plan[
        "feature_group"
    ] = (
        plan[
            "feature_group"
        ]
        .astype(str)
    )

    repository = (
        create_feature_repository(
            plan
        )
    )

    dependencies = (
        create_dependency_table(
            plan
        )
    )

    plan.to_csv(
        MASTER_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    repository.to_csv(
        FEATURE_REPOSITORY_PATH,
        index=False,
        encoding="utf-8",
    )

    dependencies.to_csv(
        DEPENDENCIES_PATH,
        index=False,
        encoding="utf-8",
    )

    plan.loc[
        plan[
            "feature_group"
        ] == "Deterministic"
    ].to_csv(
        DETERMINISTIC_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    plan.loc[
        plan[
            "feature_group"
        ] == "Domain"
    ].to_csv(
        DOMAIN_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    plan.loc[
        plan[
            "feature_group"
        ] == "Statistical"
    ].to_csv(
        STATISTICAL_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    plan.loc[
        plan[
            "feature_group"
        ] == "Machine Learning"
    ].to_csv(
        ML_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    plan.loc[
        plan[
            "feature_group"
        ] == "Inferential"
    ].to_csv(
        INFERENCE_PLAN_PATH,
        index=False,
        encoding="utf-8",
    )

    MARKDOWN_PATH.write_text(
        create_markdown_report(
            plan,
            repository,
        ),
        encoding="utf-8",
    )

    HTML_PATH.write_text(
        create_html_report(
            plan,
            repository,
        ),
        encoding="utf-8",
    )

    group_counts = (
        plan[
            "feature_group"
        ]
        .value_counts()
        .reindex(
            group_order,
            fill_value=0,
        )
    )

    summary = f"""FEATURE ENGINEERING PLAN SUMMARY
======================================================================

Training observations:
{len(data)}

Target variable:
{TARGET_COLUMN}

Total proposed feature assets:
{len(plan)}

Feature groups:
{group_counts.to_string()}

Automatically approved or implementation-ready:
{int(plan["analyst_decision"].astype(str).str.startswith("APPROVE").sum())}

Proposals requiring evaluation or approval:
{int(~plan["analyst_decision"].astype(str).str.startswith("APPROVE").sum())}

Outputs:
{OUTPUT_DIR}

Important:
This stage creates a plan and feature repository only.
It does not transform the training, validation, or test datasets.
======================================================================
"""

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)

    print("\nFEATURE PLAN BY GROUP")

    print(
        plan[
            [
                "feature_name",
                "feature_group",
                "feature_subtype",
                "creation_status",
                "used_by_machine_learning",
                "used_by_inference",
                "requires_training_fit",
                "priority",
                "analyst_decision",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOpen the HTML plan with:")

    print(
        "start results\\feature_engineering_plan\\"
        "feature_engineering_plan.html"
    )


if __name__ == "__main__":
    main()
