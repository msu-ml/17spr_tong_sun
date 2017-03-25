#rm
rm(list=ls())

#parameter
trait='height'
pheno_file='height.csv'
geno_file='BGData.RData'
trn_file='training.txt'

#library
library(BGData)

#working dir
setwd('/mnt/research/quantgen/projects/sunmeng2/proj01/input')
load.BGData(geno_file)
trn=scan(trn_file)
Y=read.table(pheno_file,sep=',',header=TRUE,stringsAsFactors=FALSE)
setwd('/mnt/research/quantgen/projects/sunmeng2/proj01/output')

#geno
X=DATA@geno
IDx=rownames(X)

#id in trn, id in x
Y=Y[Y$IID%in%trn,]
tmp=match(IDx,Y$IID)

IDy=Y$IID[tmp]
y=Y[,paste0('adjusted_',trait)][tmp]

#missing values
Y=data.frame(IID=IDy,y=y,stringsAsFactors=FALSE)
whichRows=which(!is.na(Y$y))

#gwas
gwas=GWAS(y~1,data=BGData(pheno=Y,geno=X,map=data.frame()),
          verbose=TRUE,nCores=6, i=whichRows)

#save
save(gwas,file=paste0('gwas_',trait,'.RData'))

#Mahanttan plot
nsnp=nrow(gwas)
pos=seq(1,nsnp,by=1)
adjust_p=as.vector(-log10(gwas[,4]))
chr=DATA@map$chromosome
D=cbind(pos, chr, adjust_p)

mini=aggregate(pos~chr,data=D,min) 
maxi=aggregate(pos~chr,data=D,max)

at=round((mini+maxi)/2)

#save
save(D,at,file='gwas_height_plot.RData')

quit(save='no')