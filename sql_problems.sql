' probelm 3 : Given a table of sales, find the transactions that are outliers,
  defined as being more than 3 standard deviations above the mean sales for a particular product.'
with temp_temp as (
  select stddev(amount) over(partition by  product_id) as std,
  sales_id,
  product_id,
  amount_id 
  from sales 
  ) 
select sales_id ,
product_id,
amount_id
from temp_temp
where std > 3 ;

-- 2nd solution 

with temp_temp as (
  select product_id,
  avg(amount) as avg_sales,
  stddev(amount) as std
from sales
group by product_id 
  ) 
select s.sales_id,
s.product_id, s.amount 
from sales s
join temp_temp t on t.product_id = s.product_id 
where s.amount > t.avg_sales + s.std * 3 ; 
