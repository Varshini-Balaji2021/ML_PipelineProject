# EDA Analytical Notebook

## Purpose of EDA

EDA answers five questions:

1. What does the data look like?
2. Which variables remain candidates?
3. What relationships and patterns exist?
4. What challenges may affect modelling?
5. How should modelling proceed?

## 1. Candidate Explanatory Variables

| feature | semantic_type | business_rationale | expected_relationship | candidate_status |
| --- | --- | --- | --- | --- |
| longitude | spatial_coordinate | Geographic position may capture regional market structure and nonlinear spatial effects. | Expected regional and nonlinear spatial variation. | RETAIN AS CANDIDATE |
| latitude | spatial_coordinate | Geographic position may capture regional market structure and nonlinear spatial effects. | Expected regional and nonlinear spatial variation. | RETAIN AS CANDIDATE |
| housing_median_age | discrete_numeric | The variable may have a nonlinear or piecewise relationship with house value. | Possible nonlinear or threshold effect. | RETAIN AS CANDIDATE |
| total_rooms | count | The variable may contribute through scale, density, ratio, or interaction effects. | Ambiguous raw-scale effect; ratios may be more informative. | RETAIN AS CANDIDATE |
| total_bedrooms | count | The variable may contribute through scale, density, ratio, or interaction effects. | Ambiguous raw-scale effect; ratios may be more informative. | RETAIN AS CANDIDATE |
| population | count | The variable may contribute through scale, density, ratio, or interaction effects. | Possible density and congestion effect. | RETAIN AS CANDIDATE |
| households | count | The variable may contribute through scale, density, ratio, or interaction effects. | Possible block-size and occupancy effect. | RETAIN AS CANDIDATE |
| median_income | continuous_numeric | The variable remains conceptually relevant and should be assessed jointly with other predictors. | Expected positive association with house value. | RETAIN AS CANDIDATE |
| ocean_proximity | nominal_categorical | Systematic differences between categories may explain house-price variation. | Expected systematic differences across categories. | RETAIN AS CANDIDATE |

## 2. Multicollinearity Screening

| feature_1 | feature_2 | correlation | absolute_correlation | eda_interpretation | machine_learning_implication | inferential_implication |
| --- | --- | --- | --- | --- | --- | --- |
| longitude | latitude | -0.9247 | 0.9247 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| total_rooms | total_bedrooms | 0.9293 | 0.9293 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| total_rooms | population | 0.8532 | 0.8532 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| total_rooms | households | 0.9174 | 0.9174 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| total_bedrooms | population | 0.8739 | 0.8739 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| total_bedrooms | households | 0.9797 | 0.9797 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |
| population | households | 0.9026 | 0.9026 | High pairwise association | Retain initially. Predictive models may use both variables if validation performance benefits. | Do not remove at EDA stage. Assess VIF, condition number, estimability, and theory before coefficient interpretation. |

## 3. LOWESS Deviation and Analyst Review

| feature | lowess_deviation_score | visual_evidence | analyst_decision | provisional_model_implication | figure_file |
| --- | --- | --- | --- | --- | --- |
| longitude | 0.3953 | Review the corresponding LOWESS figure | Not reviewed | Do not rely only on a linear coefficient; compare nonlinear OLS and GAM specifications. | lowess_longitude.png |
| latitude | 0.3599 | Review the corresponding LOWESS figure | Not reviewed | Do not rely only on a linear coefficient; compare nonlinear OLS and GAM specifications. | lowess_latitude.png |
| total_rooms | 0.2166 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_total_rooms.png |
| total_bedrooms | 0.1965 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_total_bedrooms.png |
| households | 0.1953 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_households.png |
| housing_median_age | 0.1945 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_housing_median_age.png |
| population | 0.1818 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_population.png |
| median_income | 0.1646 | Review the corresponding LOWESS figure | Not reviewed | Evaluate nonlinear polynomial and GAM effects. | lowess_median_income.png |

The automated LOWESS screen is provisional. Visual review, RESET,
diagnostics, validation, and test evidence are required.

## 4. Automatically Generated Variables

| recommended_feature | formula | business_meaning | recommendation_type | implementation_rule | caution |
| --- | --- | --- | --- | --- | --- |
| rooms_per_household | total_rooms / households | Average housing-space availability per household. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| bedrooms_per_room | total_bedrooms / total_rooms | Internal composition of the housing stock. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| population_per_household | population / households | Average household occupancy. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| bedrooms_per_household | total_bedrooms / households | Bedroom availability per household. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| rooms_per_person | total_rooms / population | Housing-space availability per person. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| bedrooms_per_person | total_bedrooms / population | Bedroom availability per person. | DETERMINISTIC RATIO | Create automatically | Protect against zero denominators and fit all model-dependent transformations on training data only. |

## 5. Exploratory Interaction Hypotheses

| recommended_feature | formula | business_meaning | recommendation_type | implementation_rule | caution |
| --- | --- | --- | --- | --- | --- |
| longitude_latitude_interaction | longitude × latitude | Exploratory interaction representing joint location. | EXPLORATORY INTERACTION | Evaluate as a candidate, not as a mandatory feature | Protect against zero denominators and fit all model-dependent transformations on training data only. |
| income_ocean_proximity_interaction | median_income × ocean_proximity indicators | Tests whether income effects differ by coastal category. | EXPLORATORY INTERACTION | Evaluate as a candidate, not as a mandatory feature | Protect against zero denominators and fit all model-dependent transformations on training data only. |

## 6. Business Interpretation

| feature_or_pattern | observed_pattern | possible_explanation | business_implication | inferential_caution |
| --- | --- | --- | --- | --- |
| median_income | Correlation with the response = 0.6879. | The variable may capture purchasing power, local demand, neighbourhood quality, access to services, amenities, and employment opportunities. | Useful for segmentation, pricing, location assessment, affordability analysis, and scenario design. | The relationship is associative and may overlap with location and omitted neighbourhood characteristics. |
| latitude and longitude | Weak marginal linear associations, strong joint association, and visually curved response patterns. | The outcome may vary across regional markets rather than following one uniform geographic gradient. | Supports regional segmentation, territory planning, location strategy, and geographically differentiated decision rules. | Separate linear coefficients may not adequately capture spatial structure or local dependence. |
| ocean_proximity | Highest mean response category: ISLAND; lowest: INLAND. | The categories may capture amenity value, scarcity, accessibility, and geographic market segmentation. | Useful for differentiated valuation, pricing bands, portfolio design, and location-based decisions. | Category differences may overlap with income, coordinates, and unmeasured neighbourhood characteristics. |
| housing stock and occupancy variables | Room, bedroom, household, and population counts are strongly associated with one another. | Raw counts partly measure the size of the observational unit. Ratios may better represent density, crowding, composition, and resource availability. | Supports occupancy analysis, capacity assessment, neighbourhood comparison, and more interpretable indicators. | VIF and estimability must be assessed before interpreting their coefficients separately. |

## 7. Variable Selection Strategy

Schema-approved variables → EDA candidate set → Feature generation →
Model-based assessment → Validation → Final model.

## 8. Potential Modelling Challenges

| challenge_group | challenge | evidence | implication |
| --- | --- | --- | --- |
| Statistical issue | Response asymmetry | Response skewness = 0.9857. | Consider transformation only when it improves validation or specification adequacy. |
| Data issue | Missing predictor values | Missing values remain in: total_bedrooms | Fit imputation using training data only. |
| Statistical issue | High predictor association | 7 predictor pair(s) have /r/ >= 0.80. | Run VIF and condition-number checks for inference; retain for ML unless validation indicates otherwise. |
| Statistical issue | Possible nonlinear relationships | longitude, latitude, total_rooms, total_bedrooms, households, housing_median_age, population, median_income | Compare linear, nonlinear polynomial, and smooth specifications; treat the LOWESS screen as provisional. |
| Data issue | Potential extreme observations | total_rooms, total_bedrooms, households, population, median_house_value | Retain for prediction unless domain evidence supports removal; perform sensitivity analysis for inference. |
| Statistical issue | Potential omitted variables | The available predictors may not fully measure local quality, accessibility, institutional conditions, or market expectations. | Avoid causal claims and document the limits of the available data when interpreting coefficients. |
| Statistical issue | Possible interaction effects | Geographic, socioeconomic, and categorical variables may modify one another's relationships with the response. | Evaluate selected interactions as hypotheses and retain them only when theoretically meaningful and empirically supported. |
| Data issue | Measurement limitations | Several variables are aggregated summaries rather than individual-level measurements. | Interpret findings at the observational-unit level and avoid individual-level conclusions. |
| Business / structural issue | Spatial heterogeneity | Coordinates exhibit strong joint geographic structure. | Purely linear spatial effects may be insufficient; regional segmentation or smooth spatial effects may matter. |

## 9. Model-Development Recommendations

| sequence | analytical_branch | stage | recommendation | reason |
| --- | --- | --- | --- | --- |
| 1 | Both | Candidate explanatory variables | Begin with all schema-approved predictors. | EDA is a screening stage. Weak marginal correlation does not justify exclusion. |
| 2 | Both | Feature engineering | Create deterministic ratios and evaluate selected interaction hypotheses. | Ratios may separate block size from housing density and occupancy conditions. |
| 3 | Machine Learning | Training and validation | Retain the complete engineered feature set initially; train, tune, validate, freeze, and test. | Predictive contribution may arise through joint, nonlinear, and interaction effects. |
| 4 | Inferential | Linear baseline | Estimate Linear OLS and assess VIF and condition number. | 7 high-correlation predictor pair(s) were identified. |
| 5 | Inferential | Specification decision | Apply RESET. If the linear model passes, proceed to diagnostics and robust inference. | A valid linear model should be retained when specification and diagnostics are acceptable. |
| 6 | Inferential | Alternative specification | If Linear OLS fails, evaluate nonlinear polynomial OLS and then GAM. | Exploratory curvature appears in: longitude, latitude, total_rooms, total_bedrooms, households, housing_median_age, population, median_income. |
| 7 | Inferential | Model-family evaluation | Diagnose each valid family, compare models, validate, test, and then estimate the selected model on the full sample. | Model choice should reflect specification adequacy, diagnostics, interpretability, and out-of-sample evidence. |

## 10. Executive Summary

```text
EXECUTIVE EDA SUMMARY
======================================================================

The exploratory analysis suggests that:

✓ All schema-approved predictors should remain candidates at this stage.

✓ Weak marginal correlation is not treated as evidence that a predictor
  is unimportant. Joint, nonlinear, interaction, categorical, and spatial
  contributions remain possible.

✓ 7 highly associated predictor pair(s) require
  VIF and condition-number assessment before inferential coefficient
  interpretation.

✓ The provisional LOWESS screen indicates material curvature for:
  longitude, latitude, total_rooms, total_bedrooms, households, housing_median_age, population, median_income.
  These findings require visual and model-based confirmation.

✓ 6 automatically generated ratio variable(s) have direct
  business meaning, while 2 interaction hypothesis(es)
  should be evaluated rather than imposed.

✓ Potential extreme observations should generally be retained for machine
  learning and examined through sensitivity analysis for inference.

Recommended modelling direction:

Machine Learning
    Candidate variables → Feature generation → Training → Validation
    → Tuning → Frozen model → Test → Operational use

Inferential Analysis
    Linear baseline → Multicollinearity assessment → RESET
    → Diagnostics or alternative nonlinear specification
    → Model comparison → Validation → Test
    → Full-sample estimation → Interpretation

EDA guides the next stage but does not determine the final model.
======================================================================

```

## Methodological Caution

EDA guides model development but does not establish causality, statistical
significance, coefficient estimability, or out-of-sample performance.
