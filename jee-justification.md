# JEE Data Pipeline Documentation

## Pipeline Overview

This pipeline preprocesses the JEE dataset to prepare it for machine learning modeling. It includes categorical encoding, outlier detection and treatment, robust feature scaling, and missing value imputation. These transformations ensure that the dataset is numerically clean, standardized, and suitable for downstream algorithms.

---

## Pipeline Diagram

![jee_pipeline.png](https://raw.githubusercontent.com/jjjilot/CS423/refs/heads/main/jee_pipeline.png)

---

## Step-by-Step Design Choices

### 1. Family Income Mapping (`map_family_income`)
- **Transformer**: `CustomMappingTransformer('family_income', {'Low': 0, 'Mid': 1, 'High': 2})`
- **Design Choice**: Ordinal encoding of family income  
- **Rationale**:  
  - Preserves the natural order of economic status  
  - Enables numerical treatment in models without inflating dimensionality  

---

### 2. Peer Pressure Level Mapping (`map_peer_pressure`)
- **Transformer**: `CustomMappingTransformer('peer_pressure_level', {'Low': 0, 'Medium': 1, 'High': 2})`
- **Design Choice**: Ordinal encoding of peer pressure  
- **Rationale**:  
  - Encodes increasing levels of psychological pressure numerically  
  - Maintains order structure while enabling quantitative modeling  

---

### 3. Admission Taken Mapping (`map_admission_taken`)
- **Transformer**: `CustomMappingTransformer('admission_taken', {'No': 0, 'Yes': 1})`
- **Design Choice**: Binary encoding  
- **Rationale**:  
  - Converts categorical outcome into a numeric binary feature  
  - Simplifies downstream processing and model compatibility  

---

### 4. Outlier Treatment for Daily Study Hours (`tukey_study_hours`)
- **Transformer**: `CustomTukeyTransformer(target_column='daily_study_hours', fence='outer')`
- **Design Choice**: Tukey outer fence for outlier detection  
- **Rationale**:  
  - Identifies and treats extreme values  
  - Preserves reasonable variance while improving feature reliability  

---

### 5. Study Hours Scaling (`scale_study_hours`)
- **Transformer**: `CustomRobustTransformer(target_column='daily_study_hours')`
- **Design Choice**: Robust scaling  
- **Rationale**:  
  - Normalizes values based on median and IQR  
  - Reduces the influence of remaining outliers on model training  

---

### 6. Outlier Treatment for JEE Main Score (`tukey_main_score`)
- **Transformer**: `CustomTukeyTransformer(target_column='jee_main_score', fence='outer')`
- **Design Choice**: Tukey outer fence  
- **Rationale**:  
  - Treats abnormal examination scores  
  - Improves statistical consistency across records  

---

### 7. Main Score Scaling (`scale_main_score`)
- **Transformer**: `CustomRobustTransformer(target_column='jee_main_score')`
- **Design Choice**: Robust scaling  
- **Rationale**:  
  - Ensures comparability of values  
  - Minimizes distortion from score-based anomalies  

---

### 8. Outlier Treatment for JEE Advanced Score (`tukey_advanced_score`)
- **Transformer**: `CustomTukeyTransformer(target_column='jee_advanced_score', fence='outer')`
- **Design Choice**: Tukey outer fence  
- **Rationale**:  
  - Removes or limits the impact of extreme advanced scores  
  - Enhances stability in predictive modeling  

---

### 9. Advanced Score Scaling (`scale_advanced_score`)
- **Transformer**: `CustomRobustTransformer(target_column='jee_advanced_score')`
- **Design Choice**: Robust scaling  
- **Rationale**:  
  - Protects against skew in the score distribution  
  - Centers and scales scores for model compatibility  

---

### 10. Outlier Treatment for Class 12 Percent (`tukey_12_percent`)
- **Transformer**: `CustomTukeyTransformer(target_column='class_12_percent', fence='outer')`
- **Design Choice**: Tukey outer fence  
- **Rationale**:  
  - Detects and adjusts anomalous academic performance values  
  - Improves the integrity of the feature  

---

### 11. Class 12 Percent Scaling (`scale_12_percent`)
- **Transformer**: `CustomRobustTransformer(target_column='class_12_percent')`
- **Design Choice**: Robust scaling  
- **Rationale**:  
  - Standardizes feature values using IQR  
  - Ensures the feature is comparable across all rows  

---

### 12. Imputation (`impute`)
- **Transformer**: `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice**: KNN-based imputation  
- **Rationale**:  
  - Fills missing values using similarity between records  
  - Maintains feature relationships better than simple mean or median  

---

## Pipeline Execution Order Rationale

1. **Categorical encoding**: Transforms string-based features to numerical representations early for compatibility  
2. **Outlier treatment**: Identifies and adjusts extreme values before scaling  
3. **Scaling**: Normalizes numerical ranges to avoid scale bias  
4. **Imputation**: Applied last so KNN has access to fully transformed data  

---

## Performance Considerations

- **Ordinal mappings**: Enable ordered categorical features to be used numerically  
- **Tukey fences**: Remove only extreme outliers while preserving meaningful variation  
- **Robust scaling**: Mitigates the effects of skew and outliers  
- **KNN imputation**: Preserves structural relationships between features, improving model generalization  
