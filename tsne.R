library(Rtsne)
library(umap)

## R code for k means and visualization of encoded values
## This gets changed a lot, and maybe won't run as is.
encoded = read.csv("encodings/13k_validation_encodings_1024.csv", header = F)
encoded = as.matrix(encoded)

encoded_pca = prcomp(encoded, scale = T, rank = 500)

encoded_subset = encoded[1:250,]

pdf("heatmap_1024_centered.pdf")
heatmap(t(encoded_subset), col = colorRampPalette(c("blue", "white", "red"))(256),
	hclustfun = function(d){return(hclust(d, "ward.D"))}, scale = "row")
dev.off()

set.seed(87)
clusters = kmeans(encoded, 6, iter.max = 100)
clus_color = c("blue", "salmon", "forestgreen", "black",  "purple", "cyan")

tsne = Rtsne(encoded_pca$x)
plot(tsne$Y[1:500,1], tsne$Y[1:500,2], col = clus_color[clusters$cluster[1:500]])

write.table(clusters$cluster, "encoding_1024_13k_clusters.txt" , quote = F, row.names = F, col.names = F)
read.csv("python_clusters.txt", header = F, stringsAsFactors = F)

set.seed(87)
umap_ = umap(encoded) 
svg("umap_clusters_1024_centered.svg")
plot(umap_$layout[,1], umap_$layout[,2], col = clus_color[clusters$cluster],
    ylab = "UMAP 2", xlab = "UMAP 1")
dev.off()
