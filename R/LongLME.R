library(nlme)
library(ggplot2)
library(merTools)
library(lme4)
library(broom)
library(dplyr)
setwd('E:/brain/subtype/src')

info <- read.csv('./table/AD_Cog_Long.csv')


info$cluster = info$cluster +1
info$cluster <- relevel(as.factor(info$cluster), "1")
columns = c('MMSE','ADAS11','ADNI_EF','ADNI_LAN','ADNI_VS','ADNI_MEM','ADAS13')
#info = subset(info,VIS_M < 61)
colors = c("#374E55FF","#DF8F44FF","#00A1D5FF","#B24745FF")
columns = c('MMSE')
for (column in columns){
  print(column)
  
  cinfo <- subset(info,Item == column)
  cinfo$Gender = as.factor(cinfo$Gender)
  mf <- lm(Change ~ Age + Gender + VIS_M + cluster + cluster*VIS_M,data = cinfo)
  
  mf %>% tidy %>% write.csv(file = paste0('./table/',column,"_AD_ST1.tidy.csv"))
  # Amf %>% glance %>% write.csv(file = paste0('./table/',column,"AD.glance.csv"))
  newdat <- expand.grid(Gender=unique(cinfo$Gender),
                        Age=c(min(cinfo$Age),
                              max(cinfo$Age)),
                        VIS_M = c(min(cinfo$VIS_M),
                                  max(cinfo$VIS_M)),
                        cluster = unique(cinfo$cluster),
                        PTID = unique(cinfo$PTID)
  )
  
  #geom_point(size=1,alpha=0.5)
  p <- ggplot(cinfo, aes(x=VIS_M, y=predict(mf), colour=cluster, fill=cluster)) +  
    geom_smooth( method = 'lm',se=0.95)+
    scale_color_manual(values = colors)+
    scale_fill_manual(values = colors)+
    theme_classic()+
    theme(legend.position = 'None')+
    xlab('Follow-up in months') + ylab(column) +
    scale_x_continuous(breaks=seq(0,120,24))
  print(p)
  ggsave(paste0('./draw_ABETA/ADNI_Long_',column,'.png'),width = 2,height = 2,dpi = 300)
}




p <- ggplot(newdat, aes(x=VIS_M, y=predict(mf), colour=cluster)) +
  geom_line() + 
  theme_bw(base_size=22)
print(p)

p <- ggplot(cinfo,aes(x=VIS_M , y = Change, colour = cluster)) + geom_jitter() +
  scale_color_manual(values = c("#4878d0","#ee854a","#6acc64","#d65f5f"))
print(p)




