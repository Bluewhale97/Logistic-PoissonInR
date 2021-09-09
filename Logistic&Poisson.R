#1. situations that is unreasonable to assume that the dependent variable is normally distributed(or even continuous)

#the outcome variable may be categorical

#binary variables and polytomous variable clearly arent normally distributed

#the outcome variable may be a count(for example, number of traffic accidents in a week, number of drinnks per day). Such variables take on a limited number of values and are never negative. Additionally, their mean and variance are often related(which isnt true for normally distrbuted variables)

#2. generalized linear models extend the linear-model framework to include dependent variables that are decidely non-normal

glm() #use to estimate them, we will focus on two popular models: logistic regression(where the dependent variable is categorical) and poisson regression(where the dependent variable is a count variable)

#question to solve:
#a. what personal, demographic, and relationship variables predict marital infidelity? In this case, the outcome variable is binary
#b. what impact does a drug treatment for seizures have on the number of seizures experienced over an eight-week period? In this case, the outcome variable is a count(number of seizures)

#we will use logitic regression to address the first question and poison regression to address the second

#3. generalized linear models are typically fit in R through the glm() functin
#the form of the function is similar to lm() but includes additional parameters

glm(formula, family=family(link=function),data=)#where the probability distribution(family) and cirresponding default link function(function) are given below:

binomial #link="logit"
gaussian #link="identity"
gamma #link="inverse"
inverse.gaussion #link="1/mu^2"
poisson #link="log"
quasi#link="identity", variance="constant"
quasibinomial #link="logit"
quasipoisson #link="log"


#the glm() function allows to fit a number of popular models, including logistic regression, poisson regressin and survival analysis(not considered here)

#logistic regression is applied to situations in which the response variable is dichotomous(0 or 1). The model asumes that Y follows a binomial distribution and that you can fit a linear model of the form
glm(Y~X1+X2+X3, family=binomial(link="logit"), data=mydata)

#poisson regression is applied to situationsin which the response variable is the number of events to occur in a given period of time
#the Poisson regresson model assumes that Y follows a poisson distribution and that we can fit a linear model of the form

glm(Y~X1+X2+X3, family=poisson(link="log"),data=mydata)


#its worth noting that the standrad linear model is also a special case of the generalized linear model
#if we let the link function g(uY)=uY or the identity function and specify that
#the probability distribution is normal(Gaussian), then 
glm(Y~X1+X2+X3, family=gaussian(link="identity"),data=mydata)

#would produce the same result as 
lm(Y~X1+X2+X3, data=mydata)

#the parameter estimates are derived via maximum likelihood rather than least squares

#4. supporting functions

summary() #displays detailed results for the fitted model
coefficients(), coef() #lists the model parameters(intercept and slopes) for the fitted model
confint() #provides confidence intervals for the model parameters(95% by default)
residuals()#lists the residual values in a fitted model
anova()#generates an ANOVA  table comparing two fitted models 
olot() #generates disgnostic plots for evaluating the fit of a model
predict() #uses a fitted model to predict response values for a new dataset
deviance()#deviance for the fitted model
df.residual()#residual degress of freefom for the fitted model

#5. model fit and regression diagnostics

#generally, plot predicted values expressed in the metric of the original response variable against residuals of the deviance type

#a common diagnostic plot would be

plot(predict(model, type="response"),
     residuals(model, type="deviance")) #where model is the object returned by the glm() function

#the hat values, studentized residuals and Cook's D statistics that R provides will be approximate values
#there is no general consensus on cutoff values for identifying problematic observations
#values have to be judged relative to each other

#one approach is to create index pots for each statistic and look for unusually large values

plot(hatvalues(model))
plot(rstudent(model))
plot(cooks.distance(model))

#alternatively, we could use 
library(car)
influencePlot(mopdel)#to create one omnibus plot. the horizontal axis is the leverage, the vertical axis is the studentized residual and the plotted symbol is proportional to the Cook's distance


#diagnostic plots tend to be most helpful when the response variable takes on many values. When the response variable can only take on a limited number of values(for example, logistic regression), the utility of these plots is decresed


#6. logistic regression 
#useful when predicting a binary outcome from a set of continuous and/or categorical predictor variables

install.packages("AER")
library(AER)

data(Affairs, package="AER")
summary(Affairs)

table(Affairs$affairs)
#we can see that the 52% of respondents were female, that 72% had children and that median  age for the sample was 32 years

#we are interested in whether here is in binary outcome(had an affair/didnt have an affair)
#we could transform affairs into a dichotomous factor called ynaffair with:
Affairs$ynaffair[Affairs$affairs>0]<-1

Affairs$ynaffair[Affairs$affairs==0]<-0

Affairs$ynaffair <-factor(Affairs$ynaffair, levels=c(0,1), labels=c("No","Yes"))

table(Affairs$ynaffair)

#this dichotomous factor can be used as the outcome variable in a logistic regression model
fit.full <-glm(ynaffair~gender+age+yearsmarried+children+religiousness +education+occupation+rating, family=binomial(),data=Affairs)
summary(fit.full)

#from the p values for the regression coefficients we see that gender, presence of children, education and occupation may not make a significant contribution to the equation
#let's fit a second equation without them and test whether this reduced model fits the data as well
fit.reduced <-glm(ynaffair~age+yearsmarried+religiousness+rating, data=Affairs, family=binomial())
summary(fit.reduced)

#each regression coefficient in the reduced model is statistically significant(p<.05)
#because the two models are nested(fit.reduced is a subset of fit.full), we can use the anova() function to compare them
#for generalized linear models, we will want to chi-square version of this test
anova(fit.reduced, fit.full, test="Chisq")

#nonsignificant chi-square value p=.21 suggests that the reduced model with four predictors fits as well as the full model with nine predictors
#reinforcing belieft that gender, children, education and occupation dont add significantly to the prediction

#7. interpreting the model parameters

coef(fit.reduced)
#in a logistic regression, the response being modeled is the log(odds)that Y=1
#the regression coefficients give the change in log(odds) in the response for a unit change in the predictor variable, holding all other predictor variables constant

#since the log(odds) are difficult to interpret, we can exponentiate them to put the results on an odds scale

exp(coef(fit.reduced))
#now we can see the odds of an extramarital encounter are increased by a factor of 1.106 for a one-year increase in years married
#because the predictor variables cant equal 0, the intercept isn't meaningful 

#we can use the confint() function to obtain confidence intervals for the coefficients
exp(confin(fit.reduced)) #would print 95% confidence intervals for each of the coefficients on an odds scale

#for binary logistic regression, the change in the odds of the higher value on the response variable ofr an n unit change in a predictorvariable is exp(Betaj)^n
#a 10-year increase would increase the odds by a factor of 1.106^10 for married 

#8. assessing the impact of predictors on the probability of an outcome

#use predict() function to observe the impact of varying the levels of a predictor variable on the probability of the outcome

#first step is to create an artificial dataset containing the values of the predictor variables we are interested in
#then use this artificial dataset with the predict() function to predict the probabilites of the outcome event occurring for these values

testdata<-data.frame(rating=c(1,2,3,4,5),age=mean(Affairs$age), yearsmarried=mean(Affairs$yearsmarried),
                     religiousness=mean(Affairs$religiousness))
testdata


#use the test dataset and prediction equation to obtain probabilities
testdata$prob <-predict(fit.reduced, newdata=testdata, type="response")
testdata
#the prob of an extramarital affair decreases from .53 when the marriage israted 1=very unhappy to .15 when the marriage is rated 5=very happy(holding age,years, married and religiousness constant)

#now look at the impact of age
testdata<-data.frame(rating=mean(Affairs$rating),
                     age=seq(17,57,10),
                     yearsmarried=mean(Affairs$yearsmarried),
                     religiousness=mean(Affairs$religiousness))
testdata

testdata$prob <-predict(fit.reduced, newdata=testdata, type="response"
                        testdata)

#as age increases from 17 to 57, the prob of an extramarital encounter decreases from .34 to .11, holding the other variables constant

#9. overdispersion
#the expected variance for data drawn from a binomial distribution is np(1-p)

#overdispersion occurs when the observed variance of the response variable is larger than what would be expected from a binomial distribution
#overdispersion can lead to distorted test standard errors and inaccurate tests of significance

#when over dispersion is present, we can still fit a logistic regression using the glm() function

#one way to detect overdispersion is to compare the residual deviance with residual degrees of freedom in our binomial model
#if it is considerably larger than 1, we have evidence of overdispersion

deviance(fit.reduced)/df.residual(fit.reduced)
#it is close to 1, suggesting no overdispersion


#the other approach is to  fit the model twice
#use family="binomial" and in the second instance using family="quasibinomial" .
#if glm() object returned in the first case is called fit and the object returned in the second case is called fit.od, then

pchisq(summary(fit.od)$dispersion*fit$df.residual,
       fit$df.residual, lower=F)




fit<-glm(ynaffair~age+yearsmarried+religiousness+rating, family=binomial(), data=Affairs)
fit.od <-glm(ynaffair~age+yearsmarried+religiousness+rating, family=quasibinomial(), data=Affairs)
pchisq(summary(fit.od)$dispersion*fit$df.residual,
       fit$df.residual, lower=F)
#the resulting p valie .34 is clearly not significant(p>.05), strengthning our belief that overdispersion isnt a problem


#10. extensions of logistic regression
#robust logistic regression, the glmRob() function in the robust package can be used, it is helpful when fitting logistic regression models to data containing outliers and influential observations
#multinomial logistic regression, if the response variable has more than two unordered categories, we can fit a polytomous logistic regression using mlogit() function in the mlogit package
#ordinal logistic regression, if the response variable is a set of ordered categories, we can fit an ordinal logistic regression using the lrm() function in the rms package

#ability to model a response variable with multiple categories both ordered and unordered is an important extension, but it comes at 
#the expense of greater interpretive complexity
#assessing model fit and regression diagnostics in these cases will also be more complex


#11. poisson regression
#useful when predicting an outcome variable representing counts from a set of continuous and/or categorical predictor variables

#a comprehensive yet accessible introduction to Poisson regression is provided by Coxe, West, and Aiken(2009)

install.packages("robust")
library(robust)
data(breslow.dat, package="robust")
names(breslow.dat)
summary(breslow.dat[c(6,7,8,10)])


#although there are 12 variables in the dataset, we are limiting our attention to the 4 described.
#both the baseline and post-randomization number of seizures are highly skewed 

opar <-par(no.readonly=T)
par(mfrow=c(1,2))
attach(breslow.dat)
hist(sumY, breaks=20, xlab="Seizure Count",
     main="Distribution of Seizures")
boxplot(sumY ~Trt, xlab="Treatment", main="Group Conparisons")
par(opar)

#clearly see the skewed nature of the dependent variable and the possible presence of outliers
fit <-glm(sumY~Base+Age+Trt, data=breslow.dat, family=poisson())
summary(fit)

#the output provides the deviances, regression parameters, and standard errors and tests that these parameters are 0
#note that the each of the predictor variables is significant at the p<.05

#12. interpreting the model parameters of poisson 

coef() #model coefficients are obtained, or by examining the coefficients table in the summary() function

coef(fit)
#the regression parameter .0227 for age indicates that a one year increase in age is associated with a .03 increase in the log mean number of seizures, holding baseline seizures and treatment condition constant
#intercept isnt meaningful in this case

#exponentiate the coefficients to interpret the regression cofficients 
exp(coef(fit))

#one year increase in age multiplies the expected number of seizures by 1.023, holding others constant, maening that increased age is associated with higher numbers of seizures

#13. overdispersion

#overdispersion occurs in poisson regression when the observed variance of the response variable is larger than would be predicted by the poisson distribution
#because overdispersion is often encountered when dealing with count data and can have a negative impact on the interpretation of the results

#why overdispersion sometimes occurs
#a, important predictor variable can lead to overdispersion
#overdispersion can also be caused by a phenomenon known as state dependence
#the longitudinal studies, overdispersion can be caused by the clustering inherent in repeated measures data


#qcc package provides a test for overdispersion in the Poisson case

install.packages("qcc")
library(qcc)
qcc.overdispersion.test(breslow.dat$sumY, type="poisson")
#strongly suggestng the presence of overdispersion

#we also can fit a model to data using two family like "poisson" and "quasipoisson" doing so to verify the overdispersion


#14, extentions  of poisson regression
#poisson regression with varying time period

#add offset option in the glm() function
fit<-glm(sumY~Base +Age +Trt, data=breslow.dat, offset=log(time), family=poisson)
#where sumY is the number of seizures that occurred post-randomization for a patient during the time the patient was studied
#assuming the rate doesnt vary overtime(for example, 2 seizures in 4 days in equivalent to 10 seizures in 20days)
#rate is the outcome variable

#15. zero-inflated poisson regression
#when the number of zero counts in a dataset is larger than would be predicted by the Poisson model. This can occur when there is a subgroup of the population that would never engage in the behavior being counted

#we can analyze such case by an approach using zero-inflated Poisson regression
#the approach ifts two models simultaneously, one that predicts who whould or would not have an affair, and the second that predicts how many affairs a participant would have if you excluded the permanently faithful

zeroinfl()#in the pscl package for zero-inflated poisson regression

structurual zeros#whatrever the level of parameters is, no change happened to the outcome variables


#Robust poisson regression
glmRob() #use to fit a robust generalized linear model, including robust poisson regression
#helpful in the presence of outliers and influential observations


#16. other sources
#short introduction to the generalized linear models: Dunteman and Ho(2006)
#the classic(and advanced)text on generalized linear models is procided by McCullagn and Nelder(1989)
#comprehensive and accessible presentations are provided by Dobson and Barnett(2008), Fox(2008). Faraway(2006) and Fix(2002)

