#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install statsmodels


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# To display all the columns of dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
import seaborn as sns


# # Bondora Data Preprocessing 
# 
# In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform  ([**Bondora**](https://www.bondora.com/en/public-reports#dataset-file-format)).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between **1st March 2009** and **27th January 2020**. The data
# comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.
# 
# In this notebook we will preprocess the raw dataset and will create new preprocessed csv that can be used for building credit risk models.

# In[4]:


df=pd.read_csv('Bondora_raw.csv',low_memory=False)


# In[5]:


df.shape


# In[6]:


df['Status'].value_counts()


# In[7]:


112-37


# In[8]:


df.head()


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

# In[9]:


credit_missing=df


# In[10]:


# To show all the rows of pandas dataframe
credit_missing.isnull().sum()/(len(credit_missing))*100


# Removing all the features which have more than 40% missing values

# In[11]:


# removing the columns having more than 40% missing values
credit_missing=credit_missing.drop(['ContractEndDate', 'NrOfDependants', 'EmploymentPosition',
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
       'ActiveLateLastPaymentCategory'],axis=1)


# In[12]:


# drop missing  values columns )
credit_missing.shape

Apart from missing value features there are some features which will have no role in default prediction like 'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth' (**because age is already present**), 'BiddingStartedOn','UserName','NextPaymentNr','NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension',
'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare','IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther' (**As Total income is already present which is total of all these income**), 'LoanApplicationStartedDate','ApplicationSignedHour',
       'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate',
       'LastPaymentOn', 'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn',
       'ProbabilityOfDefault', 'PrincipalOverdueBySchedule',
       'StageActiveSince', 'ModelVersion','WorseLateCategory'
# In[13]:


credit_missing.columns


# In[14]:


cols_del = ['ReportAsOfEOD', 'LoanId', 'LoanNumber', 'ListedOnUTC', 'DateOfBirth',
       'BiddingStartedOn','UserName','NextPaymentNr',
       'NrOfScheduledPayments','IncomeFromPrincipalEmployer', 'IncomeFromPension',
       'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther','LoanApplicationStartedDate','ApplicationSignedHour',
       'ApplicationSignedWeekday','ActiveScheduleFirstPaymentReached', 'PlannedInterestTillDate',
       'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn',
       'ProbabilityOfDefault', 'PrincipalOverdueBySchedule',
       'StageActiveSince', 'ModelVersion','WorseLateCategory','ExistingLiabilities','RefinanceLiabilities','DebtToIncome', 'FreeCash', 'MonthlyPaymentDay','BidsPortfolioManager','BidsApi', 'BidsManual','LoanDate', 'FirstPaymentDate', 'MaturityDate_Original','MaturityDate_Last','Amount','County','Rating','PrincipalPaymentsMade','InterestAndPenaltyPaymentsMade','PrincipalBalance','InterestAndPenaltyBalance','PreviousRepaymentsBeforeLoan','City']


# In[15]:


credit_redundant = credit_missing.drop(cols_del,axis=1)


# In[16]:


credit_redundant.shape


# In[17]:


credit_redundant.columns


# In[18]:


credit_data_duplicates=credit_redundant.drop_duplicates()


# In[19]:


credit_data_duplicates.shape


# In[20]:


credit_data_duplicates.dtypes


# ## Creating Target Variable
# 
# Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values **current, Late and repaid**. There is no default feature but there is a feature **default date** which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining **Status** and **Default date** features for creating target  variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null.
# So we will first filter out all the current status records because they are not matured yet they are current loans. 

# In[21]:


# let's find the counts of each status categories 
credit_data_duplicates.Status.value_counts()


# In[22]:


# filtering out Current Status records
credit_filtering=credit_data_duplicates[credit_data_duplicates.Status != 'Current']


# In[23]:


credit_filtering.Status.value_counts()


# In[24]:


credit_filtering.shape


# Now, we will create new target variable in which 0 will be assigned when default date is null means borrower has never defaulted while 1 in case default date is present.

# df['Status'] = df['Defaultdate'].apply(lambda x: 1 if not pd.isnull(x) else 0)

# In[25]:


def status(x):
    if(pd.isnull(x)==True):
        return 0
    else: 
        return 1


# In[26]:


credit_filtering['status']=credit_filtering.DefaultDate.apply(status)
credit_filtering['status'].unique()


# In[27]:


# check the counts of default and nondefault 
credit_filtering.status.value_counts()


# In[28]:


# let's drop the status columns
credit_target=credit_filtering.drop(['DefaultDate','Status','LastPaymentOn'],axis=1)


# In[29]:


credit_target.head()


# In[30]:


credit_target.shape


# Now, we will remove Loan Status and default date as we have already created target variable with the help of these two features

# ## Removing outliers for the numerical data

# In[31]:


credit_data_dropnull=credit_target


# In[32]:


df_2 = credit_data_dropnull[['AppliedAmount']]
ax = sns.boxplot(data=df_2)


# In[33]:


Q1_appamo = credit_data_dropnull['AppliedAmount'].quantile(0.25)
print(Q1_appamo)
Q3_appamo=credit_data_dropnull['AppliedAmount'].quantile(0.75)
print(Q3_appamo)
iqr_appamo=Q3_appamo-Q1_appamo
print(iqr_appamo)


# In[34]:


ul_appamo = Q3_appamo+1.5*iqr_appamo
print(ul_appamo)
ll_appamo = Q1_appamo-1.5*iqr_appamo
print(ll_appamo)


# In[35]:


credit_data_outliers=credit_data_dropnull.drop(credit_data_dropnull[ (credit_data_dropnull.AppliedAmount > ul_appamo) | (credit_data_dropnull.AppliedAmount < ll_appamo) ].index)


# In[36]:


df_2 = credit_data_dropnull[['IncomeTotal']]
ax = sns.boxplot(data=df_2)


# In[37]:


Q1_income = credit_data_outliers['IncomeTotal'].quantile(0.25)
print(Q1_income)
Q3_income=credit_data_outliers['IncomeTotal'].quantile(0.75)
print(Q3_income)
iqr_income=Q3_income-Q1_income
print(iqr_income)


# In[38]:


ul_income = Q3_income+1.5*iqr_income
print(ul_income)
ll_income = Q1_income-1.5*iqr_income
print(ll_income)


# In[39]:


credit_data_outliers_1=credit_data_outliers.drop(credit_data_outliers[ (credit_data_outliers.IncomeTotal > ul_income) | (credit_data_outliers.IncomeTotal < ll_income) ].index)


# In[40]:


df_2 = credit_data_dropnull[['AmountOfPreviousLoansBeforeLoan']]
ax = sns.boxplot(data=df_2)


# In[41]:


Q1_loan = credit_data_outliers_1['AmountOfPreviousLoansBeforeLoan'].quantile(0.25)
print(Q1_loan)
Q3_loan=credit_data_outliers_1['AmountOfPreviousLoansBeforeLoan'].quantile(0.75)
print(Q3_loan)
iqr_loan=Q3_loan-Q1_loan
print(iqr_loan)


# In[42]:


ul_loan = Q3_loan+1.5*iqr_loan
print(ul_loan)
ll_loan = Q1_loan-1.5*iqr_loan
print(ll_loan)


# In[43]:


credit_data_outliers_final=credit_data_outliers_1.drop(credit_data_outliers_1[ (credit_data_outliers_1.AmountOfPreviousLoansBeforeLoan > ul_loan) | (credit_data_outliers_1.AmountOfPreviousLoansBeforeLoan < ll_loan) ].index)


# In[44]:


df_3 = credit_data_dropnull[['LiabilitiesTotal']]
ax = sns.boxplot(data=df_3)


# In[45]:


Q1_liab = credit_data_outliers_final['LiabilitiesTotal'].quantile(0.25)
print(Q1_liab)
Q3_liab=credit_data_outliers_final['LiabilitiesTotal'].quantile(0.75)
print(Q3_liab)
iqr_liab=Q3_liab-Q1_liab
print(iqr_liab)


# In[46]:


ul_liab = Q3_liab+1.5*iqr_liab
print(ul_liab)
ll_liab = Q1_liab-1.5*iqr_liab
print(ll_liab)


# In[47]:


credit_data_outliers_final_up=credit_data_outliers_final.drop(credit_data_outliers_final[ (credit_data_outliers_final.LiabilitiesTotal > ul_liab) | (credit_data_outliers_final.LiabilitiesTotal < ll_liab) ].index)


# In[48]:


credit_target.shape


# In[49]:


credit_data_outliers_final_up.shape


# In[50]:


credit_target_up=credit_data_outliers_final_up


# ## checking datatype of all features
# In this step we will see any data type mismatch

# In[51]:


credit_data_duplicates.dtypes


# - First we will delete all the features related to date as it is not a time series analysis so these features will not help in predicting target variable.
# - As we can see in numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc.
# - So we will convert these features to categorical features

# In[52]:


credit_target_up.VerificationType = credit_target_up.VerificationType.astype('category')
credit_target_up.LanguageCode = credit_target_up.LanguageCode.astype('category')
credit_target_up.Gender = credit_target_up.Gender.astype('category')
credit_target_up.UseOfLoan = credit_target.UseOfLoan.astype('category')
credit_target_up.Education = credit_target_up.Education.astype('category')
credit_target_up.MaritalStatus = credit_target_up.MaritalStatus.astype('category')
credit_target_up.EmploymentStatus = credit_target_up.EmploymentStatus.astype('category')
credit_target_up.OccupationArea = credit_target_up.OccupationArea.astype('category')
credit_target_up.HomeOwnershipType = credit_target_up.HomeOwnershipType.astype('category')
credit_target_up.PreviousEarlyRepaymentsCountBeforeLoan = credit_target_up.PreviousEarlyRepaymentsCountBeforeLoan.astype('category')
credit_target_up.HomeOwnershipType = credit_target_up.HomeOwnershipType.astype('category')


# In[53]:


credit_target_up.dtypes


# Checking distribution of categorical variables

# In[54]:


sns.barplot(x = 'NewCreditCustomer', y = 'IncomeTotal', data = credit_target_up)


# In[55]:


sns.countplot(data=credit_target_up,x='NewCreditCustomer',hue='status')


# In[56]:


sns.barplot(x = 'Gender', y = 'IncomeTotal', data = credit_target_up)


# In[57]:


credit_target_up=credit_target_up[credit_target_up.Gender != 2.0]


# In[58]:


credit_target_up.Gender.value_counts()


# In[59]:


sns.countplot(data=credit_target_up,x='Gender',hue='status')


# In[60]:


sns.barplot(x = 'Country', y = 'IncomeTotal', data = credit_target_up)


# In[61]:


sns.barplot(x = 'Country', y = 'IncomeTotal', data = credit_target_up)


# checking distribution of all numeric columns

# In[62]:


visual_sample= credit_target_up.sample(n = 25000)


# In[63]:


sns.histplot(data=visual_sample,x='AppliedAmount')


# In[64]:


sns.histplot(data=visual_sample,x='Interest')


# In[65]:


sns.histplot(data=visual_sample,x='IncomeTotal')


# In[66]:


sns.histplot(data=visual_sample,x='LiabilitiesTotal')


# Now we will check the distribution of different categorical variables

# In[67]:


#1 Estonian 2 English 3 Russian 4 Finnish 5 German 6 Spanish 9 Slovakian
def lang(x):
    if(x==1):
        return 'Estonian'
    elif x==2: 
        return 'English'
    elif x==3:
        return 'Russian'
    elif x==4:
        return 'Finnish'
    elif x==6:
        return 'Spanish'
    elif x==9:
        return 'Slovakian'
    else:
        return 'Other'
credit_target_up['LanguageCode']=credit_target_up.LanguageCode.apply(lang)
credit_target_up['LanguageCode'].unique()


# In[68]:


credit_target_up.LanguageCode.value_counts()


# As we can see from above in language code w ehave only descriptions for values 1,2,3,4,5,6, and 9 but it has other values too like 21,22,15,13,10 and 7 but they are very less it may happen they are local language codes whose decription is not present so we will be treated all these values as others

# In[69]:


sns.countplot(data=credit_target_up,x='LanguageCode',hue='status')


# ## UseOfLoan
# 0 Loan consolidation 1 Real estate 2 Home improvement 3 Business 4 Education 5 Travel 6 Vehicle 7 Other 8 Health 101 Working capital financing 102 Purchase of machinery equipment 103 Renovation of real estate 104 Accounts receivable financing 105 Acquisition of means of transport 106 Construction finance 107 Acquisition of stocks 108 Acquisition of real estate 109 Guaranteeing

# As we can see from above stats most of the loans are -1 category whose description is not avaialble in Bondoro website so we have dig deeper to find that in Bondora most of the loans happened for which purpose so we find in Bondora [Statistics Page](https://www.bondora.com/en/public-statistics) most of the loans around 34.81% are for Not set purpose. so we will encode 0 as Not set category even 104,101,107,106,108,110,102 are least number of values so we are making those things as others category

# In[70]:


credit_target_up.UseOfLoan.value_counts()


# In[70]:


def loan(x):
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
credit_target_up['UseOfLoan']=credit_target_up.UseOfLoan.apply(loan)
credit_target_up['UseOfLoan'].unique()


# In[71]:


plt.figure(figsize=(20,10))
sns.countplot(data=credit_target_up,x='UseOfLoan',hue='status')


# Education
# 1 Primary education 2 Basic education 3 Vocational education 4 Secondary education 5 Higher education

# In[72]:


credit_target_up.Education.value_counts()


# -1 and 0 are less values so we remove those rows 

# In[73]:


indexeducation = credit_target_up[ (credit_target_up['Education'] == 0.0) ].index


# In[74]:


indexeducation


# In[75]:


credit_target_up.drop(indexeducation , inplace=True)


# In[76]:


indexeducation_ = credit_target_up[ (credit_target_up['Education'] == -1.0) ].index


# In[77]:


credit_target_up.drop(indexeducation_ , inplace=True)


# In[78]:


credit_target_up.Education.value_counts()


# In[79]:


def education(x):
    if(x==1):
        return 'Primary education'
    elif x==2: 
        return 'Basic education'
    elif x==3:
        return 'Vocational education'
    elif x==4:
        return 'Secondary education'
    else:
        return 'Higher education'
credit_target_up['Education']=credit_target_up.Education.apply(education)
credit_target_up['Education'].unique()


# In[80]:


plt.figure(figsize=(20,10))
sns.countplot(data=credit_target_up,x='Education',hue='status')


# 1 Married 2 Cohabitant 3 Single 4 Divorced 5 Widow

# Again Marital status of value 0 and -1 has no description so we will encode them as Not_specified

# In[81]:


credit_target_up.MaritalStatus.value_counts()


# In[82]:


def marital(x):
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
        return 'not specified'
credit_target_up['MaritalStatus']=credit_target_up.MaritalStatus.apply(marital)
credit_target_up['MaritalStatus'].unique()


# In[83]:


plt.figure(figsize=(20,10))
sns.countplot(data=credit_target_up,x='MaritalStatus',hue='status')


# 1 Unemployed 2 Partially employed 3 Fully employed 4 Self-employed 5 Entrepreneur 6 Retiree

# In[71]:


credit_target_up.EmploymentStatus.value_counts()


# In[72]:


def emp(x):
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
        return 'not specified'
credit_target_up['EmploymentStatus']=credit_target_up.EmploymentStatus.apply(emp)
credit_target_up['EmploymentStatus'].unique()


# In[73]:


plt.figure(figsize=(20,10))
sns.countplot(data=credit_target_up,x='EmploymentStatus',hue='status')


# In[74]:


credit_redundant.Restructured


# In[75]:


sns.countplot(data=credit_target_up,x='Restructured',hue='status')


# In[76]:


credit_target_up.OccupationArea.value_counts()


# 1 Other 2 Mining 3 Processing 4 Energy 5 Utilities 6 Construction 7 Retail and wholesale 8 Transport and warehousing 9 Hospitality and catering 10 Info and telecom 11 Finance and insurance 12 Real-estate 13 Research 14 Administrative 15 Civil service & military 16 Education 17 Healthcare and social help 18 Art and entertainment 19 Agriculture, forestry and fishing

# In[77]:


def occ(x):
    if(x==-1):
        return 'Not_specified'
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
credit_target_up['OccupationArea']=credit_target_up.OccupationArea.apply(occ)
credit_target_up['OccupationArea'].unique()


# In[78]:


plt.figure(figsize=(32,20))
sns.countplot(data=credit_target_up,x='OccupationArea',hue='status')


#  HomeOwnershipType
# 0 Homeless 1 Owner 2 Living with parents 3 Tenant, pre-furnished property 4 Tenant, unfurnished property 5 Council house 6 Joint tenant 7 Joint ownership 8 Mortgage 9 Owner with encumbrance 10 Other

# In[79]:


credit_target_up.HomeOwnershipType.value_counts()


# 0 Homeless 1 Owner 2 Living with parents 3 Tenant, pre-furnished property 4 Tenant, unfurnished property 5 Council house 6 Joint tenant 7 Joint ownership 8 Mortgage 9 Owner with encumbrance 10 Other

# In[80]:


def home(x):
    if(x==1):
        return 'Owner'
    elif x==2: 
        return 'Living with parents'
    elif x==3:
        return 'Tenant with furnished'
    elif x==8:
        return 'Mortgage'
    elif x==4:
        return 'Tenant without furnished'
    elif x==7:
        return 'Joint ownership'
    elif x==6:
        return 'Joint tenant'
    elif x==5:
        return 'Council house'
    else:
        return 'Other'
credit_target_up['HomeOwnershipType']=credit_target_up.HomeOwnershipType.apply(home)
credit_target_up['HomeOwnershipType'].unique()


# In[81]:


credit_target_up.HomeOwnershipType.value_counts()


# In[82]:


plt.figure(figsize=(32,20))
sns.countplot(data=credit_target_up,x='HomeOwnershipType',hue='status')


# In[83]:


credit_target_up.head()


# In[84]:


# save the final data
credit_target_up.to_csv('Data_preprocessing.csv',index=False)


# In[ ]:





# In[ ]:




