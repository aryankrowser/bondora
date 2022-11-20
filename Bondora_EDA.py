#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# To display all the columns of dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns


# # Bondora Data Preprocessing 
# 
# In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format)).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between **1st March 2009** and **27th January 2020**. The data
# comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.
# 
# In this notebook we will preprocess the raw dataset and will create new preprocessed csv that can be used for building credit risk models.

# In[3]:


credit_risk = pd.read_csv("Bondora_raw.csv",low_memory=False)


# In[4]:


credit_risk.shape


# In[5]:


credit_risk['Status'].value_counts()


# In[6]:


credit_risk.head()


# ## Data Understanding

# | Feature                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# |----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | ActiveLateCategory                     | When a loan is in Principal Debt then it will be categorized by Principal Debt days                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | ActiveLateLastPaymentCategory          | Shows how many days has passed since last payment and categorised if it is overdue                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# | ActiveScheduleFirstPaymentReached      | Whether the first payment date has been reached according to the active schedule                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | Age                                    | The age of the borrower when signing the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | Amount                                 | Amount the borrower received on the Primary Market. This is the principal balance of your purchase from Secondary Market                                                                                                                                                                                                                                                                                                                                                                                            |
# | AmountOfPreviousLoansBeforeLoan        | Value of previous loans                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | AppliedAmount                          | The amount borrower applied for originally                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | AuctionBidNumber                       | Unique bid number which is accompanied by Auction number                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
# | AuctionId                              | A unique number given to all auctions                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | AuctionName                            | Name of the Auction, in newer loans it is defined by the purpose of the loan                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | AuctionNumber                          | Unique auction number which is accompanied by Bid number                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
# | BidPrincipal                           | On Primary Market BidPrincipal is the amount you made your bid on. On Secondary Market BidPrincipal is the purchase price                                                                                                                                                                                                                                                                                                                                                                                           |
# | BidsApi                                | The amount of investment offers made via Api                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | BidsManual                             | The amount of investment offers made manually                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | BidsPortfolioManager                   | The amount of investment offers made by Portfolio Managers                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | BoughtFromResale_Date                  | The time when the investment was purchased from the Secondary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | City                                   | City of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | ContractEndDate                        | The date when the loan contract ended                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | Country                                | Residency of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | County                                 | County of the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | CreditScoreEeMini                      | 1000 No previous payments problems 900 Payments problems finished 24-36 months ago 800 Payments problems finished 12-24 months ago 700 Payments problems finished 6-12 months ago 600 Payment problems finished < 6 months ago 500 Active payment problems                                                                                                                                                                                                                                                          |
# | CreditScoreEsEquifaxRisk               | Generic score for the loan applicants that do not have active past due operations in ASNEF; a measure of the probability of default one year ahead; the score is given on a 6-grade scale: AAA (“Very low”), AA (“Low”), A (“Average”), B (“Average High”), C (“High”), D (“Very High”).                                                                                                                                                                                                                            |
# | CreditScoreEsMicroL                    | A score that is specifically designed for risk classifying subprime borrowers (defined by Equifax as borrowers that do not have access to bank loans); a measure of the probability of default one month ahead; the score is given on a 10-grade scale, from the best score to the worst: M1, M2, M3, M4, M5, M6, M7, M8, M9, M10.                                                                                                                                                                                  |
# | CreditScoreFiAsiakasTietoRiskGrade     | Credit Scoring model for Finnish Asiakastieto RL1 Very low risk 01-20 RL2 Low risk 21-40 RL3 Average risk 41-60 RL4 Big risk 61-80 RL5 Huge risk 81-100                                                                                                                                                                                                                                                                                                                                                             |
# | CurrentDebtDaysPrimary                 | How long the loan has been in Principal Debt                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | CurrentDebtDaysSecondary               | How long the loan has been in Interest Debt                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# | DateOfBirth                            | The date of the borrower's birth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | DebtOccuredOn                          | The date when Principal Debt occurred                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | DebtOccuredOnForSecondary              | The date when Interest Debt occurred                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | DebtToIncome                           | Ratio of borrower's monthly gross income that goes toward paying loans                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | DefaultDate                            | The date when loan went into defaulted state and collection process was started                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | DesiredDiscountRate                    | Investment being sold at a discount or premium                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# | EAD1                                   | Exposure at default, outstanding principal at default                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | EAD2                                   | Exposure at default, loan amount less all payments prior to default                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | Education                              | 1 Primary education 2 Basic education 3 Vocational education 4 Secondary education 5 Higher education                                                                                                                                                                                                                                                                                                                                                                                                               |
# | EL_V0                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | EL_V1                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | EL_V2                                  | Expected loss calculated by the specified version of Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | EmploymentDurationCurrentEmployer      | Employment time with the current employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | EmploymentPosition                     | Employment position with the current employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | EmploymentStatus                       | 1 Unemployed 2 Partially employed 3 Fully employed 4 Self-employed 5 Entrepreneur 6 Retiree                                                                                                                                                                                                                                                                                                                                                                                                                         |
# | ExistingLiabilities                    | Borrower's number of existing liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | ExpectedLoss                           | Expected Loss calculated by the current Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | ExpectedReturn                         | Expected Return calculated by the current Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | FirstPaymentDate                       | First payment date according to initial loan schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | FreeCash                               | Discretionary income after monthly liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# | Gender                                 | 0 Male 1 Woman 2 Undefined                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | GracePeriodEnd                         | Date of the end of Grace period                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | GracePeriodStart                       | Date of the beginning of Grace period                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | HomeOwnershipType                      | 0 Homeless 1 Owner 2 Living with parents 3 Tenant, pre-furnished property 4 Tenant, unfurnished property 5 Council house 6 Joint tenant 7 Joint ownership 8 Mortgage 9 Owner with encumbrance 10 Other                                                                                                                                                                                                                                                                                                              |
# | IncomeFromChildSupport                 | Borrower's income from alimony payments                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | IncomeFromFamilyAllowance              | Borrower's income from child support                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | IncomeFromLeavePay                     | Borrower's income from paternity leave                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | IncomeFromPension                      | Borrower's income from pension                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# | IncomeFromPrincipalEmployer            | Borrower's income from its employer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | IncomeFromSocialWelfare                | Borrower's income from social support                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | IncomeOther                            | Borrower's income from other sources                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | IncomeTotal                            | Borrower's total income                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | Interest                               | Maximum interest rate accepted in the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | InterestAndPenaltyBalance              | Unpaid interest and penalties                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | InterestAndPenaltyDebtServicingCost    | Service cost related to the recovery of the debt based on the interest and penalties of the investment                                                                                                                                                                                                                                                                                                                                                                                                              |
# | InterestAndPenaltyPaymentsMade         | Note owner received loan transfers earned interest, penalties total amount                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | InterestAndPenaltyWriteOffs            | Interest that was written off on the investment                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | InterestLateAmount                     | Interest debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | InterestRecovery                       | Interest recovered due to collection process from in debt loans                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | LanguageCode                           | 1 Estonian 2 English 3 Russian 4 Finnish 5 German 6 Spanish 9 Slovakian                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | LastPaymentOn                          | The date of the current last payment received from the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | LiabilitiesTotal                       | Total monthly liabilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | ListedOnUTC                            | Date when the loan application appeared on Primary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | LoanDate                               | Date when the loan was issued                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | LoanDuration                           | Current loan duration in months                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
# | LoanId                                 | A unique ID given to all loan applications                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | LoanNumber                             | A unique number given to all loan applications                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# | LoanStatusActiveFrom                   | How long the current status has been active                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# | LossGivenDefault                       | Gives the percentage of outstanding exposure at the time of default that an investor is likely to lose if a loan actually defaults. This means the proportion of funds lost for the investor after all expected recovery and accounting for the time value of the money recovered. In general, LGD parameter is intended to be estimated based on the historical recoveries. However, in new markets where limited experience does not allow us more precise loss given default estimates, a LGD of 90% is assumed. |
# | MaritalStatus                          | 1 Married 2 Cohabitant 3 Single 4 Divorced 5 Widow                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# | MaturityDate_Last                      | Loan maturity date according to the current payment schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | MaturityDate_Original                  | Loan maturity date according to the original loan schedule                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | ModelVersion                           | The version of the Rating model used for issuing the Bondora Rating                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | MonthlyPayment                         | Estimated amount the borrower has to pay every month                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | MonthlyPaymentDay                      | The day of the month the loan payments are scheduled for The actual date is adjusted for weekends and bank holidays (e.g. if 10th is Sunday then the payment will be made on the 11th in that month)                                                                                                                                                                                                                                                                                                                |
# | NewCreditCustomer                      | Did the customer have prior credit history in Bondora 0 Customer had at least 3 months of credit history in Bondora 1 No prior credit history in Bondora                                                                                                                                                                                                                                                                                                                                                            |
# | NextPaymentDate                        | According to schedule the next date for borrower to make their payment                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | NextPaymentNr                          | According to schedule the number of the next payment                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | NextPaymentSum                         | According to schedule the amount of the next payment                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | NoOfPreviousLoansBeforeLoan            | Number of previous loans                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
# | note_id                                | A unique ID given to the investments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | NoteLoanLateChargesPaid                | The amount of late charges the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | NoteLoanTransfersInterestAmount        | The amount of interest the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | NoteLoanTransfersMainAmount            | The amount of principal the note has received                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | NrOfDependants                         | Number of children or other dependants                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | NrOfScheduledPayments                  | According to schedule the count of scheduled payments                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | OccupationArea                         | 1 Other 2 Mining 3 Processing 4 Energy 5 Utilities 6 Construction 7 Retail and wholesale 8 Transport and warehousing 9 Hospitality and catering 10 Info and telecom 11 Finance and insurance 12 Real-estate 13 Research 14 Administrative 15 Civil service & military 16 Education 17 Healthcare and social help 18 Art and entertainment 19 Agriculture, forestry and fishing                                                                                                                                      |
# | OnSaleSince                            | Time when the investment was added to Secondary Market                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | PenaltyLateAmount                      | Late charges debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
# | PlannedInterestPostDefault             | The amount of interest that was planned to be received after the default occurred                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | PlannedInterestTillDate                | According to active schedule the amount of interest the investment should have received                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | PlannedPrincipalPostDefault            | The amount of principal that was planned to be received after the default occurred                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# | PlannedPrincipalTillDate               | According to active schedule the amount of principal the investment should have received                                                                                                                                                                                                                                                                                                                                                                                                                            |
# | PreviousEarlyRepaymentsBeforeLoan      | How much was the early repayment amount before the loan                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
# | PreviousEarlyRepaymentsCountBeforeLoan | How many times the borrower had repaid early                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | PreviousRepaymentsBeforeLoan           | How much the borrower had repaid before the loan                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | PrincipalBalance                       | Principal that still needs to be paid by the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | PrincipalDebtServicingCost             | Service cost related to the recovery of the debt based on the principal of the investment                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | PrincipalLateAmount                    | Principal debt amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | PrincipalOverdueBySchedule             | According to the current schedule, principal that is overdue                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | PrincipalPaymentsMade                  | Note owner received loan transfers principal amount                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | PrincipalRecovery                      | Principal recovered due to collection process from in debt loans                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | PrincipalWriteOffs                     | Principal that was written off on the investment                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | ProbabilityOfDefault                   | Probability of Default, refers to a loan’s probability of default within one year horizon.                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | PurchasePrice                          | Investment amount or secondary market purchase price                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
# | Rating                                 | Bondora Rating issued by the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | Rating_V0                              | Bondora Rating issued by version 0 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | Rating_V1                              | Bondora Rating issued by version 1 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | Rating_V2                              | Bondora Rating issued by version 2 of the Rating model                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | RecoveryStage                          | Current stage according to the recovery model 1 Collection 2 Recovery 3 Write Off                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | RefinanceLiabilities                   | The total amount of liabilities after refinancing                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
# | ReScheduledOn                          | The date when the a new schedule was assigned to the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | Restructured                           | The original maturity date of the loan has been increased by more than 60 days                                                                                                                                                                                                                                                                                                                                                                                                                                      |
# | SoldInResale_Date                      | The date when the investment was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
# | SoldInResale_Price                     | The price of the investment that was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# | SoldInResale_Principal                 | The principal remaining of the investment that was sold on Secondary market                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# | StageActiveSince                       | How long the current recovery stage has been active                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
# | Status                                 | The current status of the loan application                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
# | UseOfLoan                              | 0 Loan consolidation 1 Real estate 2 Home improvement 3 Business 4 Education 5 Travel 6 Vehicle 7 Other 8 Health 101 Working capital financing 102 Purchase of machinery equipment 103 Renovation of real estate 104 Accounts receivable financing 105 Acquisition of means of transport 106 Construction finance 107 Acquisition of stocks 108 Acquisition of real estate 109 Guaranteeing obligation 110 Other business All codes in format 1XX are for business loans that are not supported since October 2012  |
# | UserName                               | The user name generated by the system for the borrower                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
# | VerificationType                       | Method used for loan application data verification 0 Not set 1 Income unverified 2 Income unverified, cross-referenced by phone 3 Income verified 4 Income and expenses verified                                                                                                                                                                                                                                                                                                                                    |
# | WorkExperience                         | Borrower's overall work experience in years                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
# | WorseLateCategory                      | Displays the last longest period of days when the loan was in Principal Debt                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | XIRR                                   | XIRR (extended internal rate of return) is a methodology to calculate the net return using the loan issued date and amount, loan repayment dates and amounts and the principal balance according to the original repayment date. All overdue principal payments are written off immediately. No provisions for future losses are made & only received (not accrued or scheduled) interest payments are taken into account.                                                                                          |

# # Percentage of Missing Values

# In[7]:


credit_risk.isnull().sum()


# In[8]:


# checking the percenatge of null values
credit_risk.isnull().sum()/(len(credit_risk))*100


# Removing all the features which have more than 40% missing values

# In[9]:


# print missing values columns 
miss_col=['ContractEndDate', 'NrOfDependants', 'EmploymentPosition',
       'WorkExperience', 'PlannedPrincipalTillDate', 'CurrentDebtDaysPrimary',
       'DebtOccuredOn', 'CurrentDebtDaysSecondary',
       'DebtOccuredOnForSecondary',
       'PlannedPrincipalPostDefault', 'PlannedInterestPostDefault', 'EAD1',
       'EAD2', 'PrincipalRecovery', 'InterestRecovery', 'RecoveryStage',
       'EL_V0', 'Rating_V0', 'EL_V1', 'Rating_V1', 'Rating_V2',
       'ActiveLateCategory', 'CreditScoreEsEquifaxRisk',
       'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEeMini',
       'PrincipalWriteOffs', 'InterestAndPenaltyWriteOffs',
       'PreviousEarlyRepaymentsBefoleLoan', 'GracePeriodStart',
       'GracePeriodEnd', 'NextPaymentDate', 'ReScheduledOn',
       'PrincipalDebtServicingCost', 'InterestAndPenaltyDebtServicingCost',
       'ActiveLateLastPaymentCategory']


# In[10]:


# drop missing  values columns )
credit_risk = credit_risk.drop(miss_col, axis=1)


# In[11]:


credit_risk.shape


# In[12]:


credit_risk['NrOfScheduledPayments'].head()


# Apart from missing value features there are some features which will have no role in default prediction like 'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth' (**because age is already present**), 'BiddingStartedOn','UserName','NextPaymentNr','NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension',
# 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther' (**As Total income is already present which is total of all these income**), 'LoanApplicationStartedDate','ApplicationSignedHour',
#        'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate',
#        'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn',
#        'ProbabilityOfDefault', 'PrincipalOverdueBySchedule',
#        'StageActiveSince', 'ModelVersion','WorseLateCategory'

# In[13]:


cols_del = ['LastPaymentOn','ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth',
       'BiddingStartedOn','UserName','NextPaymentNr',
       'NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther','LoanApplicationStartedDate','ApplicationSignedHour',
       'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate',
       'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn',
       'ProbabilityOfDefault', 'PrincipalOverdueBySchedule',
       'StageActiveSince', 'ModelVersion','WorseLateCategory','ExistingLiabilities','RefinanceLiabilities','DebtToIncome', 'FreeCash', 'MonthlyPaymentDay','BidsPortfolioManager','BidsApi', 'BidsManual','LoanDate', 'FirstPaymentDate', 'MaturityDate_Original','MaturityDate_Last','Amount','County','Rating','PrincipalPaymentsMade','InterestAndPenaltyPaymentsMade','PrincipalBalance','InterestAndPenaltyBalance','PreviousRepaymentsBeforeLoan']


# In[ ]:





# In[14]:


credit_risk_clean = credit_risk.drop(cols_del,axis=1)


# In[15]:


credit_risk_clean.shape


# In[16]:


credit_risk_clean.head(5)


# In[17]:


credit_risk_clean.columns


# In[18]:


# dropping duplicates if any
credit_risk_clean=credit_risk_clean.drop_duplicates()


# In[19]:


credit_risk_clean.shape


# ## Creating Target Variable
# 
# Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values **current, Late and repaid**. There is no default feature but there is a feature **default date** which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining **Status** and **Default date** features for creating target  variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null.
# So we will first filter out all the current status records because they are not matured yet they are current loans. 

# In[20]:


# let's find the counts of each status categories 
credit_risk_clean['Status'].value_counts()


# In[21]:


# filtering out Current Status records
credit_risk_filter =credit_risk_clean[credit_risk_clean.Status != 'Current']


# In[22]:


credit_risk_filter['Status'].value_counts()


# Now, we will create new target variable in which 0 will be assigned when default date is null means borrower has never defaulted while 1 in case default date is present.

# In[23]:


credit_risk_filter['Status'] = credit_risk_filter['DefaultDate'].apply(lambda x: 1 if not pd.isnull(x) else 0)


# In[24]:


credit_risk_filter['Status'].value_counts()


# Now, we will remove Loan Status and default date as we have already created target variable with the help of these two features

# In[25]:


# let's drop the DefaultDate column
credit_risk_filter = credit_risk_filter.drop(['DefaultDate'], axis=1)


# In[26]:


credit_risk_filter.shape


# ## checking datatype of all features
# In this step we will see any data type mismatch

# In[27]:


credit_risk_filter.dtypes


# Checking distribution of categorical variables

# In[28]:


object_cols = credit_risk_filter.select_dtypes(include="object").columns.tolist()
object_cols


# checking distribution of all numeric columns

# In[29]:


numeric_cols = credit_risk_filter.select_dtypes(include="int64").columns.tolist()
numeric_cols


# In[30]:


numeric_cols1 = credit_risk_filter.select_dtypes(include="float64").columns.tolist()
numeric_cols1


# ## Using pandas describe() to find outliers

# In[31]:


credit_risk_filter.describe()


# On comparing the mean and the max value of the columns we can understand the huge difference and remove those outlier from columns

# In[33]:


pip install plotly


# In[34]:


import plotly.express as px
fig = px.box(credit_risk_filter, y='AppliedAmount')
fig.show()


# In[35]:


fig = px.box(credit_risk_filter, y='MonthlyPayment')
fig.show()


# In[36]:


fig = px.box(credit_risk_filter, y='IncomeTotal')
fig.show()


# In[37]:


fig = px.box(credit_risk_filter, y='NoOfPreviousLoansBeforeLoan')
fig.show()


# In[38]:


fig = px.box(credit_risk_filter, y='AmountOfPreviousLoansBeforeLoan')
fig.show()


# In[39]:


# For MonthlyPayment
Q1_MonthlyPayment = credit_risk_filter['MonthlyPayment'].quantile(0.25)
print(Q1_MonthlyPayment)
Q3_MonthlyPayment=credit_risk_filter['MonthlyPayment'].quantile(0.75)
print(Q3_MonthlyPayment)
IQR_MonthlyPayment=Q3_MonthlyPayment-Q1_MonthlyPayment
print(IQR_MonthlyPayment)


# In[40]:


ul_MonthlyPayment = Q3_MonthlyPayment+1.5*IQR_MonthlyPayment
print(ul_MonthlyPayment)
ll_MonthlyPayment = Q1_MonthlyPayment-1.5*IQR_MonthlyPayment
print(ll_MonthlyPayment)


# In[41]:


credit_risk_filter=credit_risk_filter.drop(credit_risk_filter[ (credit_risk_filter.MonthlyPayment > ul_MonthlyPayment) | (credit_risk_filter.MonthlyPayment < ll_MonthlyPayment) ].index)


# In[42]:


# For AmountOfPreviousLoansBeforeLoan
Q1_AmountOfPreviousLoans = credit_risk_filter['AmountOfPreviousLoansBeforeLoan'].quantile(0.25)
print(Q1_AmountOfPreviousLoans)
Q3_AmountOfPreviousLoans=credit_risk_filter['AmountOfPreviousLoansBeforeLoan'].quantile(0.75)
print(Q3_AmountOfPreviousLoans)
IQR_AmountOfPreviousLoans=Q3_AmountOfPreviousLoans-Q1_AmountOfPreviousLoans
print(IQR_AmountOfPreviousLoans)


# In[43]:


ul_AmountOfPreviousLoans = Q3_AmountOfPreviousLoans+1.5*IQR_AmountOfPreviousLoans
print(ul_AmountOfPreviousLoans)
ll_AmountOfPreviousLoans = Q1_AmountOfPreviousLoans-1.5*IQR_AmountOfPreviousLoans
print(ll_AmountOfPreviousLoans)


# In[44]:


credit_risk_filter=credit_risk_filter.drop(credit_risk_filter[ (credit_risk_filter.AmountOfPreviousLoansBeforeLoan > ul_AmountOfPreviousLoans) | (credit_risk_filter.AmountOfPreviousLoansBeforeLoan < ll_AmountOfPreviousLoans) ].index)


# In[45]:


# for IncomeTotal
Q1_IncomeTotal = credit_risk_filter['IncomeTotal'].quantile(0.25)
print(Q1_IncomeTotal)
Q3_IncomeTotal=credit_risk_filter['IncomeTotal'].quantile(0.75)
print(Q3_IncomeTotal)
IQR_IncomeTotal=Q3_IncomeTotal-Q1_IncomeTotal
print(IQR_IncomeTotal)


# In[46]:


ul_IncomeTotal = Q3_IncomeTotal+1.5*IQR_IncomeTotal
print(ul_IncomeTotal)
ll_IncomeTotal = Q1_IncomeTotal-1.5*IQR_IncomeTotal
print(ll_IncomeTotal)


# In[47]:


credit_risk_filter=credit_risk_filter.drop(credit_risk_filter[ (credit_risk_filter.IncomeTotal > ul_IncomeTotal) | (credit_risk_filter.IncomeTotal < ll_IncomeTotal) ].index)


# - First we will delete all the features related to date as it is not a time series analysis so these features will not help in predicting target variable.
# - As we can see in numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc.
# - So we will convert these features to categorical features

# Now we will check the distribution of different categorical variables

# In[48]:


credit_risk_filter.VerificationType = credit_risk_filter.VerificationType.astype('category')
credit_risk_filter.LanguageCode = credit_risk_filter.LanguageCode.astype('category')
credit_risk_filter.Gender = credit_risk_filter.Gender.astype('category')
credit_risk_filter.UseOfLoan = credit_risk_filter.UseOfLoan.astype('category')
credit_risk_filter.Education = credit_risk_filter.Education.astype('category')
credit_risk_filter.MaritalStatus = credit_risk_filter.MaritalStatus.astype('category')
credit_risk_filter.EmploymentStatus = credit_risk_filter.EmploymentStatus.astype('category')
credit_risk_filter.OccupationArea = credit_risk_filter.OccupationArea.astype('category')
credit_risk_filter.HomeOwnershipType = credit_risk_filter.HomeOwnershipType.astype('category')
credit_risk_filter.PreviousEarlyRepaymentsCountBeforeLoan = credit_risk_filter.PreviousEarlyRepaymentsCountBeforeLoan.astype('category')


# In[49]:


sns.countplot(data=credit_risk_filter,x='Gender',hue='Status')


# In[50]:


sample= credit_risk_filter.sample(n =5000)


# In[51]:


sns.histplot(data=sample,x='Interest')


# In[52]:


sns.histplot(data=sample,x='IncomeTotal',bins=50)


# In[53]:


sns.histplot(data=sample,bins=50,x='LiabilitiesTotal')


# As we can see from above in language code w ehave only descriptions for values 1,2,3,4,5,6, and 9 but it has other values too like 21,22,15,13,10 and 7 but they are very less it may happen they are local language codes whose decription is not present so we will be treated all these values as others

# In[54]:


def lang_code(x):
    if(x==1):
        return 'Estonian'
    elif x==2: 
        return 'English'
    elif x==3:
        return 'Russian'
    elif x==4:
        return 'Finnish'
    elif x==5:
        return 'German'
    elif x==6:
        return 'Spanish'
    elif x==9:
        return 'Slovakian'
    else:
        return 'Other'
credit_risk_filter['LanguageCode']=credit_risk_filter.LanguageCode.apply(lang_code)
credit_risk_filter['LanguageCode'].unique()


# In[55]:


sns.countplot(data=credit_risk_filter,x='LanguageCode',hue='Status')


# ## UseOfLoan
# 
# 0 Loan consolidation 1 Real estate 2 Home improvement 3 Business 4 Education 5 Travel 6 Vehicle 7 Other 8 Health 101 Working capital financing 102 Purchase of machinery equipment 103 Renovation of real estate 104 Accounts receivable financing 105 Acquisition of means of transport 106 Construction finance 107 Acquisition of stocks 108 Acquisition of real estate 109 Guaranteeing
# 
# As we can see from above stats most of the loans are -1 category whose description is not avaialble in Bondoro website so we have dig deeper to find that in Bondora most of the loans happened for which purpose so we find in Bondora Statistics Page most of the loans around 34.81% are for Not set purpose. so we will encode 0 as Not set category even 104,101,107,106,108,110,102 are least number of values so we are making those things as others category

# In[56]:


credit_risk_filter.UseOfLoan.value_counts()


# In[57]:


def UseOfLoan(x):
    if(x==-1):
        return 'No Specified purpose'
    elif x==2: 
        return 'Home improvement'
    elif x==0:
        return 'Loan consolidation'
    elif x==6:
        return 'Vehicle'
    elif x==3:
        return 'Business'
    elif x==5:
        return 'Travel'
    elif x==8:
        return 'Health'
    elif x==4:
        return 'Education'
    elif x==1:
        return 'Real estate'
    else:
        return 'Other'
credit_risk_filter['UseOfLoan']=credit_risk_filter.UseOfLoan.apply(UseOfLoan)
credit_risk_filter['UseOfLoan'].unique()


# In[58]:


plt.figure(figsize=(16,8))
sns.countplot(data=credit_risk_filter,x='UseOfLoan',hue='Status')


# Again as we can see from above description for -1 and 0 in case of education is not present so we will encode them as Not_present as we dont know anything about them.

# In[59]:


credit_risk_filter.Education.value_counts()


# In[60]:


def education(x):
    if(x==1):
        return 'Primary education'
    elif x==2: 
        return 'Basic education'
    elif x==3:
        return 'Vocational education'
    elif x==4:
        return 'Secondary education'
    elif x==5:
        return 'Higher education'
    else:
        return 'Not_present'
credit_risk_filter['Education']=credit_risk_filter.Education.apply(education)
credit_risk_filter['Education'].unique()


# In[61]:


plt.figure(figsize=(16,8))
sns.countplot(data=credit_risk_filter,x='Education',hue='Status')


# In[62]:


credit_risk_filter.MaritalStatus.value_counts()


# Again Marital status of value 0 and -1 has no description so we will encode them as Not_specified

# In[63]:


def maritalStatus(x):
    if(x==1):
        return 'Married'
    elif x==2: 
        return 'Cohabitant'
    elif x==3:
        return 'Single'
    elif x==4:
        return 'Divorced'
    elif x==5:
        return 'Widow'
    else:
        return 'Not_specified'
credit_risk_filter['MaritalStatus']=credit_risk_filter.MaritalStatus.apply(maritalStatus)
credit_risk_filter['MaritalStatus'].unique()


# In[64]:


plt.figure(figsize=(16,10))
sns.countplot(data=credit_risk_filter,x='MaritalStatus',hue='Status')


# In[65]:


credit_risk_filter.EmploymentStatus.value_counts()


# In[66]:


def employment(x):
    if(x==1):
        return 'Unemployed'
    elif x==2: 
        return 'Partially employed'
    elif x==3:
        return 'Fully employed'
    elif x==4:
        return 'Self-employed'
    elif x==5:
        return 'Entrepreneur'
    elif x==6:
        return 'Retiree'
    else:
        return 'other'
credit_risk_filter['EmploymentStatus']=credit_risk_filter.EmploymentStatus.apply(employment)
credit_risk_filter['EmploymentStatus'].unique()


# In[67]:


sns.countplot(data=credit_risk_filter,x='NewCreditCustomer',hue='Status')


# In[68]:


sns.countplot(data=credit_risk_filter,x='Restructured',hue='Status')


# In[69]:


credit_risk_filter.OccupationArea.value_counts()


# In[70]:


def occupationArea(x):
    if(x==-1):
        return 'Not_specified'
    elif x==1: 
        return 'Other'
    elif x==2: 
        return 'Mining'
    elif x==3: 
        return 'Processing'
    elif x==6:
        return 'Construction'
    elif x==7:
        return 'Retail and wholesale'
    elif x==8:
        return 'Transport and warehousing'
    elif x==9:
        return 'Hospitality and catering'
    elif x==10:
        return 'Info and telecom'
    elif x==11:
        return 'Finance and insurance'
    elif x==13:
        return 'Research'
    elif x==14:
        return 'Administrative'
    elif x==15:
        return 'Civil service & military'
    elif x==16:
        return 'Education'
    elif x==17:
        return 'Healthcare and social help'
    elif x==19:
        return 'Agriculture, forestry and fishing'
    else:
        return 'Other'
credit_risk_filter['OccupationArea']=credit_risk_filter.OccupationArea.apply(occupationArea)
credit_risk_filter['OccupationArea'].unique()


# In[71]:


plt.figure(figsize=(32,20))
sns.countplot(data=credit_risk_filter,x='OccupationArea',hue='Status')


# 0 Homeless 1 Owner 2 Living with parents 3 Tenant, pre-furnished property 4 Tenant, unfurnished property 5 Council house 6 Joint tenant 7 Joint ownership 8 Mortgage 9 Owner with encumbrance 10 Other

# In[72]:


credit_risk_filter.HomeOwnershipType.value_counts()


# In[73]:


def homeOwnershipType(x):
    if(x==0):
        return 'Homeless'
    elif x==1: 
        return 'Owner'
    elif x==2: 
        return 'Living with parents'
    elif x==3: 
        return 'Tenant pre-furnished'
    elif x==4:
        return 'Tenant, unfurnished'
    elif x==5:
        return 'Council house'
    elif x==6:
        return 'Joint tenant'
    elif x==7:
        return 'Joint ownership'
    elif x==8:
        return 'Mortgage'
    elif x==9:
        return 'Owner with encumbrance'
    else:
        return 'Other'
credit_risk_filter['HomeOwnershipType']=credit_risk_filter.HomeOwnershipType.apply(homeOwnershipType)
credit_risk_filter['HomeOwnershipType'].unique()


# In[74]:


plt.figure(figsize=(32,20))
sns.countplot(data=credit_risk_filter,x='HomeOwnershipType',hue='Status')


# In[75]:


# save the final data
credit_risk_filter.to_csv('Bondora_preprocessed.csv',index=False)


# In[76]:


df=pd.read_csv('Bondora_preprocessed.csv')


# In[77]:


df.head()


# In[78]:


df.shape


# In[ ]:




