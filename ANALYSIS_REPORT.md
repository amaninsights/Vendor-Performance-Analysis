# Vendor Performance Analysis Report

## Executive Summary

This analysis examines vendor performance across key business metrics to identify optimization opportunities and strategic insights. The study encompasses 8,564 profitable product records, revealing significant patterns in sales performance, profitability, and operational efficiency.

## Analysis Objectives

The analysis was designed to address critical business questions:
- Identification of top-performing vendors by sales and purchase volume
- Assessment of underperforming high-margin products requiring promotional focus
- Evaluation of inventory turnover efficiency and stock management
- Analysis of pricing strategy effectiveness and margin optimization
- Review of freight cost structures and logistics efficiency

## Key Findings
## Summary Statistics
![Summary Statistics](/images/summary%20statistics.png)

![Summary Histogram Statistics](/images/histogram.png)
## Key Findings & Insights

### 📊 Key Observations from the Inventory Dataset
#### 🔻 Negative and Zero Values

**Gross Profit:** The minimum gross profit is $-52,002.78, indicating that certain products or transactions are being sold at a significant loss — possibly due to elevated purchase costs or aggressive discounting strategies.

**Profit Margin:** The lowest recorded profit margin is 0%, signaling instances where either revenue is zero or less than cost, leading to no profitability. These should be flagged for review.

**Total Sales Quantity & Sales Revenue:** Some records show zero values for both sales quantity and sales dollars. This suggests that certain items were procured but never sold — potentially slow-moving, obsolete, or misclassified inventory.

### 🚩 Outliers Identified by High Standard Deviations

**Purchase Price & Actual Selling Price:** Maximum values of $5,681.81 and $7,499.99 are significantly above their respective means ($24.39 and $35.64), indicating the presence of high-value or premium products. These outliers may skew average-based analysis and should be assessed separately.

**Freight Cost:** Exhibits extreme variability — ranging from $0.09 to $257,032.07. This could point to:
- Large-volume or bulk shipments
- Vendor-side logistics inefficiencies
- Improper freight allocations in data entry

**Stock Turnover Ratio:** Ranges from 0 (unsold inventory) to 274.5 (extremely fast-moving products).
A value above 1 typically suggests that sales exceed purchases, which may be explained by:
- Sales fulfilled from old inventory
- Data entry mismatches between sales and procurement timelines

### Data Filtering
To enhance the reliability of the insights, we removed incosistent data points where:
- Gross Profit <= 0(to exclude transactions leading to losses)
- Profit Margin <= 0(to ensure analysis focuses on profitable transactions)
- Sales Quantity <= 0(to eliminate inventory items that were procured but never sold)

![correlation Insight](/images/correlation.png)


**PurchasePrice Price vs. Total Sales Dollars & Gross Profit:** has weak correlations (-0.012 and -0.016), suggesting that price variations do not significantly impact sales revenue or profit. 

**Total Purchase Quantity vs. Total Sales Quantity:** Strong correlation (0.999), confirming efficient inventory turnover. 

**Profit Margin vs. Total Sales Price:** Negative correlation (-0.179) suggesting increasing sales price may lead to reduced margins, possibly due to competitive pricing pressures. 

**Stock Turnover vs. GrossProfit:** has weak negative correlation (-0.038&-0.055), indicating that faster turnover does not necessarily result in higher profitability

## Research Questions & Key Findings

1. Brands with Promotinal or Pricing Adjustments

![brands with promotional or pricing adjustments](/images/findings/q1.png)

**198 brands** exhibit lower sales but higher progit margins, which could benift from targeted marketing,promotions,or price adjustments to increase volume without compromising profit margins.

![brands with promotional or pricing adjustments](/images/findings/a1.png)

2. **Top Vendors by Sales and Purchase Contribution**

The top 10 vendors contribute 65.69% of total purchases, while the remaining vendors contribute only 34.31%. This over-reliance on a few vendors may increase supply chain risk and limit negotiation power indicating a need for diversification.

![Top Vendors and Brands Performance](/images/findings/q2.png)

3. **Impact of Bulk Purchasing on Cost Savings**

Vendors buying in large quantities recieve a 72% lower unit cost($10.78)per unit vs higher unit costs in smaller orders.

Bulk pricing strategies encourage larger orders, increasing total sales while maintaing profitability.

![Impact of Bulk Purchasing on Cost Savings](/images/findings/q3.png)

4. **Identify Vendors with Low Inventory Turnover**

Total Unsold Inventory Capital:$2.71M

Slow-moving inventory increases storage costs,reduce cash flow efficency and affects overall profitability.

Identifying vendors with low inventory turnover can help focus on inventory optimization strategies and minizing financial strain.

![Low Inventory Turnover](/images/findings/q4.jpeg)

5. **Profit Margin Comparison: High vs Low-Performing Vendors**

Top Vendor's Profit Margin(95% CI):(30.74%,31.61%), Mean:31.17%

Low Vendor's Profit Margin(95% CI):(40.48%,42.62%), Mean:41.55%

Low-performing vendors maintain higher margins but struggle with sales volumes, indicating potential pricing inefficiences or market reach issues.

Actionable Insights:

- Top-performing vendors: Optimize profitability by adjusting pricing, reducing operational costs, or offering bundled promotions.
- Low-performing vendors: Improve marketing efforts, optimize pricing strategies, and enhance distribution networks.

![Actionable Insights](/images/findings/q5.png)

6. **Statistical Validation of Profit Differences**

**Hypothesis Testing:**

**Research Question:** Is there a significant difference in profit margins between top-performing and low-performing vendors?

**H₀ (Null Hypothesis):** There is no significant difference in the mean profit margins of top-performing and low-performing vendors (μ₁ = μ₂)

**H₁ (Alternative Hypothesis):** The mean profit margins of top-performing and low-performing vendors are significantly different (μ₁ ≠ μ₂)

**Test Results:**
- **T-Statistic:** -17.6695
- **P-Value:** < 0.0001 (highly significant)
- **Significance Level:** α = 0.05

**Result:** **REJECT H₀**

**Conclusion:** There is statistically significant evidence that profit margins differ between top and low-performing vendors (p < 0.0001).

**Implication:** High margin vendors may benefit from better pricing strategies, while top-selling vendors could focus on cost efficiency.

## Final Recommendations

- Re-evaluate pricing for low-sales, high-margins brands to boost sales volume without sacrificing profitability.

- Diversify vendor partnerships to reduce dependency on a few vendors and improve supply chain resilience.

- Leverage bulk purchasing to negotiate lower unit costs and improve overall profitability.

-Optimize slow-moving inventory by adjusting pricing, marketing, or product placement to increase sales.

-Enchance marketing and distribution strategies for low-performing vendors to drive higher sales volumes without compromising profit margins.

-By Implementing these recommendations, the company can enhance its overall profitability, reduce dependency on a few vendors, and improve inventory turnover efficiency.