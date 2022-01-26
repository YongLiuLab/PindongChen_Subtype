library(devtools)
#install_github("jfortin1/CombatHarmonization/R/neuroCombat")

library(neuroCombat)
data<- read.csv("E:/brain/subtype/src/ADATA/ADNI.csv",header = TRUE)
info <- read.csv("E:/brain/subtype/src/ADATA/ADNI_INFO.csv",header=TRUE)
data <- as.matrix(data)
age <- as.numeric(info$AGE)
group <- as.factor(info$GROUP)
gender <- as.factor(info$GENDER)
batch <- as.numeric(info$batch)
mod  <- model.matrix(~age+gender+group)
combat.harmonized <- neuroCombat(dat=t(data), batch=batch, mod=mod,mean.only = TRUE)
write.table(combat.harmonized[1], "E:/brain/subtype/src/ADATA/GMV_Combat.txt",row.names = FALSE,col.names = FALSE)

dat = matrix(runif(5*10), 5, 10)
#library(lme4)
#library(lmerTest)
#library(nlme)
#for(i in seq(from=1,to=216)){
#  info$data1 = t(data[i,])
#  info$batch = as.factor(info$batch)
#  info$GENDER = as.factor(info$GENDER)
#  mf <- lme(data1 ~ AGE + GENDER, random = ~1|batch ,data = info)
#  data[i,] = data[i,] - predict(mf)
#}
#write.table(data, "E:/brain/subtype/src/ADATA/ADNI_Combat.txt",row.names = FALSE,col.names = FALSE)

#info$data1 = t(data[100,])
#info$batch = as.factor(info$batch)
#info$GENDER = as.factor(info$GENDER)
#mf <- lmer(data1 ~ AGE + GENDER +(1|batch) ,data = info)
#summary(mf)
#info$data1Pr = predict(mf)#

#p <- ggplot(info,aes(x=AGE,y=data1,colour = batch)) +
#  geom_point(size=1,alpha = 0.25) +
#  geom_point(aes(y=data1Pr),size=3,alpha=.5) +
#  geom_line(aes(y=data1Pr),size=1)+
#  theme_bw() 
  #scale_color_brewer(palette = 'Set1')
#print(p)

#p <- ggplot(subset(info,batch==31),aes(x=AGE,y=data1,colour = batch)) +
#  geom_point(size=1,alpha = 0.25) +
#  geom_point(aes(y=data1Pr),size=3,alpha=.5) +
#  geom_line(aes(y=data1Pr),size=1)+
#  theme_bw() 
#scale_color_brewer(palette = 'Set1')
#print(p)


#library(ggplot2)
#p <- ggplot(info,aes(x=batch,y=data1))
#p + geom_jitter()
#p + geom_boxplot()
