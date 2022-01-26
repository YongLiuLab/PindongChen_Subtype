library(NMF)
library(rJava)
library(xlsxjars)
library(xlsx)
library(hNMF)
path="E:/brain/subtype/src/data/"

Data<- read.csv('E:/brain/subtype/src/data/ADNI_X_ABETA.csv',header = TRUE)


estim.r <- nmf(Data, 2:6, method="nsNMF",nrun=30, seed=123456)
coph <- estim.r$measures$cophenetic
coph.diff=coph[1:length(coph)-1]-coph[2:length(coph)]
k.best=which.max(coph.diff)+1
print(k.best)
plot(2:6,coph,type="b",col="purple")
rss <- estim.r$measures$rss
rss.diff=rss[1:length(rss)-1]-rss[2:length(rss)]
plot(estim.r)
cluster4 <- nmf(data, 4, method="ns",nrun=100, seed=123456)
W <- basis(cluster4)
H <- coef(cluster4) 
write.table(W,paste0(path,"ADNI_W_ABETA.txt"),row.names = FALSE,col.names = FALSE)
write.table(H,paste0(path,"ADNI_H_ABETA.txt"),row.names = FALSE,col.names = FALSE)
heatmap(W,Rowv = NA,Colv = NA,revC = TRUE)



randomData<- matrix(runif(nrow(data)*ncol(data)),nrow(data),ncol(data))
estim.rand.r <- nmf(randomData, 2:6, method="ns",nrun=30, seed=123456)
coph <- estim.rand.r$measures$cophenetic
coph.diff=coph[1:length(coph)-1]-coph[2:length(coph)]
rss.rand <- estim.rand.r$measures$rss
rss.rand.diff=rss.rand[1:length(rss.rand)-1]-rss.rand[2:length(rss.rand)]