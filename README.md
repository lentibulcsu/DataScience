# Machine Learning for Non-Defective Software Module Identification

## 1. Project Overview & Business Case
The purpose of the project is to develop a machine-learning model to identify non-defective (clean) software modules and improve software quality. Businesses face high costs from software defects, limited testing resources, complex code that increases risk, as well as security vulnerabilities and performance problems.

Traditional approaches focus on detecting defects after they occur. This project takes a proactive stance by identifying characteristics of clean, high-quality code modules that are least likely to contain defects. By using historical data on software metrics, an AI machine-learning model can predict which modules demonstrate hallmarks of quality code—such as low cyclomatic complexity, high test coverage, minimal past defects, and few static analysis warnings.

This approach enables organizations to:
- **Benchmark Quality Standards**: Understand what distinguishes clean code from defective code, establishing data-driven quality benchmarks
- **Optimize Resource Allocation**: Focus intensive QA efforts on higher-risk modules while streamlining reviews for predicted non-defective modules
- **Reward Best Practices**: Identify and replicate patterns from clean code modules across development teams
- **Reduce False Alarms**: Rather than flagging everything for review, confidently identify modules that meet quality thresholds

The model addresses the extreme class imbalance (only ~3% of modules are non-defective) by learning the rare patterns that characterize truly clean code, enabling teams to prioritize testing efforts and reduce overall debugging costs before deployment. 

## 2. Project Scope & Deliverables
The project utilizes the existing dataset of 60k historical software module records with an extreme class imbalance (1,777 non-defective vs. 58,223 defective modules, representing only ~3% clean code).
* **In Scope:** Exploratory Data Analysis (EDA) to understand the characteristics that distinguish non-defective code modules (for example, high `test_coverage`, low `cyclomatic_complexity`, minimal `past_defects`, and few `static_analysis_warnings`).
* **In Scope:** Training and evaluating models with a focus on accurately identifying the minority class (non-defective modules) using techniques such as SMOTE for class balancing.
* **In Scope:** Feature importance analysis to identify the most predictive attributes of clean, high-quality code.
* **Out of Scope:** Automated patching or rewriting of the defective code.
* **Out of Scope:** Real-time integration into the CI/CD pipeline.

**Expected Deliverables:**
* Data Profiling & EDA Report identifying key characteristics of non-defective software modules.
* Baseline Model for non-defective code identification.
* Refined Machine Learning Models (Random Forest, XGBoost, Support Vector Machines) optimized for accurately identifying the minority class (non-defective modules).
* Feature importance analysis highlighting the strongest predictors of clean code quality.

## 3. Risks, Constraints, and Assumptions
* **Risks:** The extreme class imbalance (approx. 97% defective vs. 3% non-defective) makes it challenging for models to learn the rare patterns that characterize clean code.
* **Risks:** High False Negatives: If the model fails to identify a truly non-defective module (predicting it as defective), the team may waste QA resources on unnecessary deep reviews of already-clean code.
* **Risks:** High False Positives: If the model incorrectly predicts a defective module as "non-defective," bugs will be pushed to production with reduced scrutiny.
* **Constraints:** The project is constrained to using only the 23 existing historical features/columns available in the current dataset.
* **Assumptions:** The historical metrics (like `past_defects`, `test_coverage`, and `static_analysis_warnings`) remain stable predictors of code quality for future sprints.

## 4. Schedule and Milestones
The project is expected to start on 2026.03.01 and reach completion by 2026.06.30. The tentative milestone schedule is as follows:

* **Project kickoff & scope definition:** February 23, 2026 to February 24, 2026.
* **Data acquisition & data understanding:** February 25, 2026 to February 28, 2026.
* **Exploratory Data Analysis (EDA):** March 1, 2026 to March 6, 2026.
* **Feature engineering & data preprocessing:** March 7, 2026 to March 12, 2026.
* **Model development (supplier reliability prediction):** March 13, 2026 to March 20, 2026.
* **Model evaluation & interpretation:** March 21, 2026 to March 25, 2026.
* **Risk scoring framework & documentation:** March 26, 2026 to March 29, 2026.
* **Final review & presentation:** March 30, 2026 to March 31, 2026.

## 5. Team, Costs, and Benefits
**Project Team & Resources:**
* **Project Team:** Lead Data Scientist, Data Analyst, Project Manager.
* **Support Resources:** QA Leads, Senior Code Reviewers, DevOps Engineer.
* **Special Needs:** Compute resources for model training, access to historical code repository metrics.

**Estimated Costs ($32,900 Total):**
* **Labour:** $30,400 for Data Scientist hours ($80/hr) and QA consultation hours ($60/hr).
* **Development Cost:** $1,500 flat rate for specific tooling or cloud compute costs.
* **Hardware:** $0 for local machine upgrades.
* **Software:** $1,000 flat rate for licenses for specialized IDEs or ML cloud platforms.

**Expected Benefits ($135,000 Total):**
* **Qualitative Benefits:** Faster release cycles, reduced QA bottleneck, and better allocation of QA resources.
* **Quantitative Benefits:** QA Labor Cost Reduction totaling $135,000, based on an estimate of 2,250 QA hours saved annually at $60/hour.
