
CREATE DATABASE  Saas_Churn;

USE Saas_Churn;



-- ============================================================
-- MODULE 10 — SQL
-- 1. CREATE TABLES
-- ============================================================

CREATE TABLE Customers (
    CustomerID VARCHAR(20) PRIMARY KEY,
    CompanyName VARCHAR(150),
    Industry VARCHAR(100),
    Country VARCHAR(100),
    City VARCHAR(100),
    EmployeeCount FLOAT,
    SignupDate DATE,
    AcquisitionChannel VARCHAR(100)
);


CREATE TABLE Subscriptions (
    SubscriptionID VARCHAR(20) PRIMARY KEY,
    CustomerID VARCHAR(20),
    PlanName VARCHAR(100),
    BillingTerm VARCHAR(50),
    MRR DECIMAL(12,2),
    Seats FLOAT,
    StartDate DATE,
    EndDate DATE,
    Status VARCHAR(50)
);


CREATE TABLE Usage (
    CustomerID VARCHAR(20),
    SubscriptionID VARCHAR(20),
    [Month] DATE,
    Logins FLOAT,
    ActiveUsers FLOAT,
    FeatureUsed VARCHAR(100),
    APICalls FLOAT,
    SessionMinutes FLOAT
);


CREATE TABLE Tickets (
    TicketID VARCHAR(20) PRIMARY KEY,
    CustomerID VARCHAR(20),
    OpenedDate DATE,
    Category VARCHAR(100),
    Priority VARCHAR(50),
    ResolutionHours FLOAT,
    SatisfactionScore FLOAT
);

SELECT COUNT(*) AS Customer_Count
FROM Cleaned_saas_customers;

SELECT COUNT(*) AS Subscription_Count
FROM Cleaned_saas_subscriptions;

SELECT COUNT(*) AS Usage_Count
FROM Cleaned_saas_usage;

SELECT COUNT(*) AS Ticket_Count
FROM Cleaned_saas_tickets;




-- QUERY 1
-- TOTAL MRR BY COUNTRY

-- Q1 How much recurring revenue comes from each country?

SELECT
    c.Country,
    COUNT(DISTINCT c.CustomerID) AS Customer_Count,
    SUM(s.MRR) AS Total_MRR
FROM cleaned_saas_customers AS c
LEFT JOIN cleaned_saas_subscriptions AS s
    ON c.CustomerID = s.CustomerID
GROUP BY
    c.Country
ORDER BY
    Total_MRR DESC;

--This covers JOIN and GROUP BY
--This shows which countries contribute the most recurring revenue.

-- QUERY 2
-- PLANS WITH HIGH TOTAL MRR

--Q2 Which subscription plans generate more than a chosen MRR amount?

SELECT
    PlanName,
    COUNT(*) AS Subscription_Count,
    SUM(MRR) AS Total_MRR,
    AVG(MRR) AS Average_MRR
FROM cleaned_saas_Subscriptions
GROUP BY
    PlanName
HAVING
    SUM(MRR) > 10000
ORDER BY
    Total_MRR DESC;

--GROUP BY + HAVING is used because we are filtering an aggregate result.


-- QUERY 3
-- CUSTOMER REVENUE SEGMENT

-- Q3 How can customers be classified according to their total MRR?

SELECT
    c.CustomerID,
    c.CompanyName,
    SUM(ISNULL(s.MRR, 0)) AS Total_MRR,

    CASE
        WHEN SUM(ISNULL(s.MRR, 0)) >= 1000
            THEN 'High Revenue'

        WHEN SUM(ISNULL(s.MRR, 0)) >= 500
            THEN 'Medium Revenue'

        ELSE 'Low Revenue'
    END AS Revenue_Category

FROM cleaned_saas_Customers AS c

LEFT JOIN cleaned_saas_Subscriptions AS s
    ON c.CustomerID = s.CustomerID

GROUP BY
    c.CustomerID,
    c.CompanyName

ORDER BY
    Total_MRR DESC;

-- satisfies the CASE requirement.


-- QUERY 4
-- CUSTOMERS ABOVE AVERAGE MRR

--Q4 Which customers generate more MRR than the average customer?
--subquery

SELECT
    c.CustomerID,
    c.CompanyName,
    SUM(s.MRR) AS Total_MRR

FROM cleaned_saas_Customers AS c

INNER JOIN cleaned_saas_Subscriptions AS s
    ON c.CustomerID = s.CustomerID

GROUP BY
    c.CustomerID,
    c.CompanyName

HAVING
    SUM(s.MRR) >
    (
        SELECT AVG(Customer_MRR)
        FROM
        (
            SELECT
                CustomerID,
                SUM(MRR) AS Customer_MRR
            FROM cleaned_saas_Subscriptions
            GROUP BY CustomerID
        ) AS CustomerRevenue
    )

ORDER BY
    Total_MRR DESC;

--subquery for
--The inner query first calculates MRR per customer.
--The outer query then compares each customer's MRR against the average customer MRR.


--QUERY 5
-- CUSTOMER USAGE SUMMARY USING CTE

--Q5 Which customers have the highest product usage?
-- CTE

WITH CustomerUsage AS
(
    SELECT
        CustomerID,
        SUM(Logins) AS Total_Logins,
        SUM(ActiveUsers) AS Total_ActiveUsers,
        SUM(APICalls) AS Total_APICalls,
        SUM(SessionMinutes) AS Total_SessionMinutes

    FROM Usage

    GROUP BY
        CustomerID
)

SELECT
    c.CustomerID,
    c.CompanyName,
    cu.Total_Logins,
    cu.Total_ActiveUsers,
    cu.Total_APICalls,
    cu.Total_SessionMinutes

FROM Customers AS c

INNER JOIN CustomerUsage AS cu
    ON c.CustomerID = cu.CustomerID

ORDER BY
    cu.Total_Logins DESC;



-- QUERY 6
-- RANK CUSTOMERS BY MRR

--Q6 How do customers rank by recurring revenue?
--This uses a window function.

WITH CustomerRevenue AS
(
    SELECT
        CustomerID,
        SUM(MRR) AS Total_MRR

    FROM cleaned_saas_Subscriptions

    GROUP BY
        CustomerID
)

SELECT
    c.CustomerID,
    c.CompanyName,
    cr.Total_MRR,

    RANK() OVER (
        ORDER BY cr.Total_MRR DESC
    ) AS MRR_Rank

FROM cleaned_saas_Customers AS c

INNER JOIN CustomerRevenue AS cr
    ON c.CustomerID = cr.CustomerID

ORDER BY
    MRR_Rank;

--assigns a revenue ranking without collapsing the customer rows.

-- QUERY 7
-- MONTHLY LOGIN TREND

--Q7How does monthly login activity change compared with the previous month?
-- Second window-function example


WITH MonthlyUsage AS
(
    SELECT
        [Month],
        SUM(Logins) AS Total_Logins

    FROM cleaned_saas_Usage

    GROUP BY
        [Month]
)

SELECT
    [Month],
    Total_Logins,

    LAG(Total_Logins) OVER (
        ORDER BY [Month]
    ) AS Previous_Month_Logins,

    Total_Logins
    -
    LAG(Total_Logins) OVER (
        ORDER BY [Month]
    ) AS Login_Change

FROM MonthlyUsage

ORDER BY
    [Month];

--LAG() - to compare each month with the previous month.


-- QUERY 8
-- EXPLICIT ORPHAN CUSTOMER RECORDS

--Q8Are there usage records whose CustomerID does not exist in Customers?


SELECT
    u.CustomerID,
    COUNT(*) AS Orphan_Record_Count
FROM cleaned_saas_Usage AS u
LEFT JOIN Customers AS c
    ON u.CustomerID = c.CustomerID
WHERE
    c.CustomerID IS NULL
GROUP BY
    u.CustomerID
ORDER BY
    Orphan_Record_Count DESC;

--The LEFT JOIN keeps every usage record.
-- WHERE c.CustomerID IS NULL - selects records where no matching parent customer exists.


--Q ORPHAN SUBSCRIPTIONS

SELECT
    s.CustomerID,
    COUNT(*) AS Orphan_Subscription_Count
FROM cleaned_saas_Subscriptions AS s
LEFT JOIN Customers AS c
    ON s.CustomerID = c.CustomerID
WHERE
    c.CustomerID IS NULL
GROUP BY
    s.CustomerID;



--QWhich customers have many tickets and low satisfaction?
--This combines JOIN, GROUP BY, HAVING, and CASE.

---- SUPPORT RISK ANALYSIS

SELECT
    c.CustomerID,
    c.CompanyName,

    COUNT(t.TicketID) AS Ticket_Count,

    AVG(t.SatisfactionScore) AS Average_Satisfaction,

    CASE
        WHEN AVG(t.SatisfactionScore) < 2
             AND COUNT(t.TicketID) >= 3
            THEN 'High Support Risk'

        WHEN AVG(t.SatisfactionScore) < 3
            THEN 'Needs Attention'

        ELSE 'Normal'
    END AS Support_Category

FROM cleaned_saas_Customers AS c

INNER JOIN cleaned_saas_Tickets AS t
    ON c.CustomerID = t.CustomerID

GROUP BY
    c.CustomerID,
    c.CompanyName

HAVING
    COUNT(t.TicketID) >= 2

ORDER BY
    Average_Satisfaction ASC;