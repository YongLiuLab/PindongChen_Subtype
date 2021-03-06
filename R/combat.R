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