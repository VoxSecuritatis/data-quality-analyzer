# Data Quality Threshold Reference

| Check / Dimension        | Threshold / Rule of Thumb                           | Severity Levels                              | Notes |
|---------------------------|-----------------------------------------------------|----------------------------------------------|-------|
| **Missing Values**        | <5% (OK), 5–30% (Warning), >30% (High Risk)         | ✅ Green <5%, ⚠️ Yellow 5–30%, 🔴 Red >30%     | >30% missing often dropped unless critical. |
| **Duplicate Rows**        | >1–5% flagged                                       | ✅ Green <1%, ⚠️ Yellow 1–5%, 🔴 Red >5%       | Should be 0% for unique IDs. |
| **Outliers (Z-score)**    | Z > 3 or Z < -3                                     | Count % outliers per column                   | Standard statistical definition. |
| **Outliers (IQR)**        | Outside Q1 - 1.5*IQR or Q3 + 1.5*IQR                | Count % outliers per column                   | Robust to non-normal data. |
| **Class Imbalance**       | <80/20 OK, 80/20–90/10 Warning, >90/10 Severe       | ✅ Balanced <80/20, ⚠️ Warning 80–90/10, 🔴 Red >90/10 | Severe imbalance requires resampling. |
| **Correlation**           | |r| > 0.9 (strong correlation)                      | ✅ Green <0.85, ⚠️ Yellow 0.85–0.9, 🔴 Red >0.9 | VIF >10 also indicates multicollinearity. |
| **Cardinality (Categorical)** | Unique categories >50% of rows                  | ⚠️ Flagged                                    | Likely ID columns (e.g., user_id). |
| **Low Variance Features** | Variance ≈ 0 or >99% same value                     | 🔴 Flagged                                    | Provides no predictive power. |
| **Invalid Values**        | Violates domain/business rules (e.g., negative age) | 🔴 Flagged                                    | Domain-specific thresholds needed. |
| **Feature Scaling**       | Std dev differs >100x between numeric features      | ⚠️ Flagged                                    | Suggest normalization / scaling. |
| **Time Series Gaps**      | Missing/duplicate timestamps                        | ⚠️ Flagged                                    | Depends on expected frequency. |

# © 2025 Brock Frary. All rights reserved.