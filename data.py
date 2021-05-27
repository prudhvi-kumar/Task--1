import pandas
db = pandas.read_csv("SalaryData.csv")
print(db)
a = db["YearsExperience"]
y = db["Salary"]
import numpy
x = numpy.array(a).reshape(-1 , 1)
from sklearn.linear_model import LinearRegression
mind = LinearRegression()
mind.fit(x , y)
print(mind.predict([[input()]]))
print(mind.coef_)
print(mind.intercept_)

