#parameter
trait='height'
pheno_file='height.csv'
geno_file='BGData.RData'
trn_file='training.txt'
tst_file='testing.txt'
block_file='SUMMARIES.RData'
arrayid=as.integer(Sys.getenv("PBS_ARRAYID"))

#library
library(methods)
library(MASS)
library(BGData)
library(parallel)

#working dir
setwd('/mnt/research/quantgen/projects/sunmeng2/proj01/input')
load.BGData(geno_file)
load(block_file)
trn=scan(trn_file)
tst=scan(tst_file)
Y=read.table(pheno_file,sep=',',header=TRUE,stringsAsFactors=FALSE)
setwd('/mnt/research/quantgen/projects/sunmeng2/proj01/output')


#datasets
X=DATA@geno
Y.TRN=Y[Y$IID%in%trn,]
rowsX_trn=sort(which(rownames(X)%in%Y.TRN$IID))# we will follow the order in X and sort Y accordingly
tmp=match(rownames(X)[rowsX_trn],Y.TRN$IID)
Y.TRN=Y.TRN[tmp,]
stopifnot( all(rownames(X)[rowsX_trn]==Y.TRN$IID))
write(Y.TRN$IID,file=paste0('IDS_scores_trn_',trait,'.txt'),ncol=1)

Y.TST=Y[Y$IID%in%tst,]
rowsX_tst=sort(which(rownames(X)%in%Y.TST$IID))# we will follow the order in X and sort Y accordingly
tmp=match(rownames(X)[rowsX_tst],Y.TST$IID)
Y.TST=Y.TST[tmp,]
stopifnot( all(rownames(X)[rowsX_tst]==Y.TST$IID))
write(Y.TST$IID,file=paste0('IDS_scores_tst_',trait,'.txt'),ncol=1)

#blocks
blocks=as.integer(SUMMARIES[,'blocks'])

uniqueBlocks=unique(blocks)

blockIndex=splitIndices(length(uniqueBlocks), 10)

OUT=matrix(nrow=length(blockIndex[[arrayid]]),ncol=6)
colnames(OUT)=c('block','N-SNPs','R2','F','numdf','p-value')

OUT[,1]=uniqueBlocks[blockIndex[[arrayid]]]

y=Y.TRN[,paste0('adjusted_',trait)]

bHat=rep(0,nrow(SUMMARIES))
SNP_names=rownames(SUMMARIES)

time1=proc.time()

for(i in 1:nrow(OUT)){
  
  j=which(blocks==OUT[i,1])
  OUT[i,2]=length(j)  # of SNPs
  Z<-X[rowsX_trn,j,drop=FALSE]
  Z<-scale(Z,scale=FALSE,center=TRUE)
  # naive imputation
  TMP<-is.na(Z)
  if(any(TMP)){
    Z[TMP]=0
  }
  
  n=nrow(Z)
  D=as.data.frame(cbind(y,Z))
  
  fm0=lm(y~1,data=D)
  fm9=lm(y~.,data=D)
  fm=step(object=fm0,scope=list(lower=fm0,upper=fm9),
          k=2,direction='forward',trace=FALSE)

  estimates=fm$coef
  if(length(estimates)>1){	
    estimates=estimates[-1]
    if(ncol(Z)==1){
      tmp=j
    }else{		
      
      new=rep(NA,length(estimates))
      for (k in 1:length(estimates)){
        new[k]=substr(names(estimates)[k],2,nchar(names(estimates)[k])-1)
      }
      
      tmp=match(new,SNP_names)
    }
    bHat[tmp]=estimates
    fm=summary(fm)
    OUT[i,3]=fm$r.squared
    OUT[i,4:5]=fm$fstatistic[1:2]
    OUT[i,6]=1-pf(q=fm$fstatistic[1],df1=fm$fstatistic[2],df2=fm$fstatistic[3])
    
  }
  if(i%%1000==0){
    save(OUT,SNP_names,blocks,bHat,file=paste0('block_stepwise_gwas_',trait,'_',arrayid,'.RData'))
    print(i)
    print((proc.time()[3]-time1[3])/3600)
  }
  #print(i)
}
time2=proc.time()

temp=which(SUMMARIES[,'blocks']%in%OUT[,1])
bHat=bHat[temp]

save(OUT,SNP_names,blocks,bHat,file=paste0('block_stepwise_gwas_',trait,'_',arrayid,'.RData'))

quit(save='no')

