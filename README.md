# Machine Learning for Non-Defective Code Identification

## 1. Project Overview & Business Case
The purpose of the project is to develop a machine-learning model to improve software quality. Defects within the software prevent a business from operating at the most efficient level. Businesses face high costs of software defects, limited testing resources, complex code that increases risk, as well as security vulnerabilities and performance problems. 

By using historical data on software detection, an AI machine-learning model can predict bugs, costs, and risks more efficiently. This approach will allocate testing resources more efficiently and help reduce debugging costs. Identifying defect-prone modules early allows teams to prioritize testing efforts and improve overall software quality before deployment. 

## 2. Project Scope & Deliverables
The project utilizes the existing dataset of 60k historical software module records. 
* **In Scope:** Exploratory Data Analysis (EDA) to understand the drivers of clean code (for example, high `test_coverage`, or low `past_defects`).
* **In Scope:** Training and evaluating models with a focus on optimizing for the minority class.
* **Out of Scope:** Automated patching or rewriting of the defective code.
* **Out of Scope:** Real-time integration into the CI/CD pipeline.

**Expected Deliverables:**
* Data Profiling & EDA Report.
* Baseline Model.
* Refined Machine Learning Models (Random Forest, XGBoost, Support Vector Machines) optimized for the minority class.

## 3. Risks, Constraints, and Assumptions
* **Risks:** The extreme class imbalance (approx. 97% defective vs. 3% clean) makes it difficult for a model to learn the minority class. 
* **Risks:** High False Positives: If the model incorrectly predicts a defective module as "clean," bugs will be pushed to production.
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
