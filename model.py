import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

class Encoder(nn.Module):
    def __init__(self, n_features, n_latent=32):
        super(Encoder, self).__init__()
        # Change the output dimensions of fc6 to be 2 * n_latent
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2 * n_latent)
    #
    def forward(self, x):
        h = self.leaky_relu(self.fc1(x))
        h = self.leaky_relu(self.fc2(h))
        h = self.leaky_relu(self.fc3(h))
        h = self.leaky_relu(self.fc4(h))
        h = self.leaky_relu(self.fc5(h))
        return self.fc6(h)


class Decoder(nn.Module):
    def __init__(self, n_latent, n_covars, n_features):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_latent + n_covars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, n_features)
    #
    def forward(self, z, covars):
        #print(f"before concat - z shape: {z.shape}, covars shape: {covars.shape}")
        z_covars = torch.cat((z, covars), dim=1)
        #print(f"after concat - z_covars shape: {z_covars.shape}")
        h = self.leaky_relu(self.fc1(z_covars))
        h = self.leaky_relu(self.fc2(h))
        h = self.leaky_relu(self.fc3(h))
        h = self.leaky_relu(self.fc4(h))
        h = self.leaky_relu(self.fc5(h))
        return self.fc6(h)


class Encoder(nn.Module):
    def __init__(self, n_features, n_latent=32):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2 * n_latent)
        self.leaky_relu = nn.LeakyReLU()  # Add LeakyReLU here
    #
    def forward(self, x):
        h = self.leaky_relu(self.fc1(x))
        h = self.leaky_relu(self.fc2(h))
        h = self.leaky_relu(self.fc3(h))
        h = self.leaky_relu(self.fc4(h))
        h = self.leaky_relu(self.fc5(h))
        return self.fc6(h)


""" class Decoder(nn.Module):
    def __init__(self, n_latent, n_covars, n_features):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_latent + n_covars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, n_features)
        self.leaky_relu = nn.LeakyReLU()  # Add LeakyReLU here
    #
    def forward(self, z, covars):
        z_covars = torch.cat((z, covars), dim=1)
        h = self.leaky_relu(self.fc1(z_covars))
        h = self.leaky_relu(self.fc2(h))
        h = self.leaky_relu(self.fc3(h))
        h = self.leaky_relu(self.fc4(h))
        h = self.leaky_relu(self.fc5(h))
        return self.fc6(h)
 """

class Decoder(nn.Module):
    def __init__(self, n_latent, n_covars, n_features):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(n_latent + n_covars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, n_features)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    #
    def forward(self, z, covars):
        z_covars = torch.cat((z, covars), dim=1)
        h = self.leaky_relu(self.fc1(z_covars))
        h = self.leaky_relu(self.fc2(h))
        h = self.leaky_relu(self.fc3(h))
        h = self.leaky_relu(self.fc4(h))
        h = self.leaky_relu(self.fc5(h))
        return self.sigmoid(self.fc6(h))  # Use Sigmoid for final output



#########################################

class orthocoder(nn.Module):
    def __init__(self, n_features, n_covars, n_latent=32):
        super(orthocoder, self).__init__()
        ## USED TO HAVE JUST REAL FEATURES, BUT TRYING TO CONCAT W/INPUT
        #self.encoder = Encoder(n_features, n_latent)
        self.encoder = Encoder(n_features+n_covars, n_latent)
        self.decoder = Decoder(n_latent, n_covars, n_features)
    #
    def forward(self, x, covars):
        mu_log_var = self.encoder(x)
        mu, log_var = torch.chunk(mu_log_var, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, covars)
        return recon_x, mu, log_var
    #
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps*std

class CustomVAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomVAELoss, self).__init__()
        self.reduction = reduction
        self.epoch = 0
    #
    #
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps*std
    #
    #
    def set_epoch(self, epoch):
        self.epoch = epoch
    #
    #
    def forward(self, input_data, recon_data, mu, log_var, covariates, epoch=0):
        self.epoch=epoch
        # Reconstruction loss
        #print("recon loss")
        recon_loss = F.mse_loss(recon_data, input_data[:,:-covariates.shape[-1]], reduction=self.reduction)
        #print("\tfinished")
        # KL divergence
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # Orthogonality loss
        ortho_loss = 0
        ## calculate the average cosine similarity of the differences
        ## I don't know if this loop is the way to go or not. It doesn't evern work
        for i in range(covariates.shape[1]):
            covar = covariates[:, i]
            #print(f"mu shape: {mu.shape}")
            #print(f"covar shape: {covar.shape}")
            ## THE SINGLE LINE BELOW WORKS, JUST COMMENTED OUT FOR PROTOTYPING ABOVE
            ortho_loss += torch.mean(torch.abs(F.cosine_similarity(mu, covar.unsqueeze(-1), dim=0)))
        #for i in range(covariates.shape[1]):
        #    covar = covariates[:, i]
        #    ortho_loss += torch.mean(torch.abs(F.cosine_similarity(mu, covar.unsqueeze(0), dim=0)))
        ortho_loss /= covariates.shape[1]  # take the mean over all covariates
        # Total loss is a weighted sum of the above
        ortho_scale = 5e-2* min(1,self.epoch/20)# go from 0 up to 1 linearly over 20 epochs, then plateu at target
        #ortho_scale = 1e-3
        if torch.any(torch.rand(1) < .0005):
            print("recon_loss:",recon_loss, "kld_loss:", kld_loss ,"  ortho_loss:",ortho_scale*ortho_loss)
        return recon_loss + 0*kld_loss + ortho_scale*ortho_loss 



###################################


def min_max_scale(data):
    """
    This function scales input data between 0 and 1 using Min-Max scaling.
    """
    # make sure the data is a floating point type
    data = data.float()
    min_val = torch.min(data, dim=0)[0]
    max_val = torch.max(data, dim=0)[0]
    return (data - min_val) / (max_val - min_val)


def get_encoded_representation(model, data):
    """
    This function passes the input data through the encoder part of the VAE and 
    returns the encoded representation (after the reparameterization trick).
    """
    with torch.no_grad():  # we don't need gradients for this operation
        mu_log_var = model.encoder(data)
        mu, log_var = torch.chunk(mu_log_var, 2, dim=1)
        z = model.reparameterize(mu, log_var)
    return z




pca_nmf_torch = pca_nmf_torch.float()
covar_torch = covar_torch.float()
for i in range(covar_torch.shape[1]):
    covar_torch[:, i] = min_max_scale(covar_torch[:, i])

pca_nmf_torch=min_max_scale(pca_nmf_torch)


pca_nmf_torch.shape[1]
n_features = pca_nmf_torch.shape[1]  # Number of primary data features
n_covars = covar_torch.shape[1]  # Number of covariates
learning_rate=1e-6
n_latent = 48
model = orthocoder(n_features, n_covars, n_latent)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# RMSprop
criterion = CustomVAELoss()


# create Tensor datasets
data = TensorDataset(torch.cat((pca_nmf_torch,covar_torch),axis=1), covar_torch)

# dataloaders
batch_size = 32 # you can change the batch size depending on your setup

# make sure to SHUFFLE your data
dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)


loss_list = []
n_epochs=100
for epoch in range(n_epochs):
    for inputs, covars in dataloader:
        # Forward pass
        recon_batch, mu, log_var = model(inputs, covars)
        # Compute loss
        criterion.set_epoch(epoch)
        loss = criterion(inputs, recon_batch, mu, log_var, covars, epoch=epoch)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    loss_list.append(loss.item())


#latents = get_encoded_representation(model, pca_nmf_torch)
latents = get_encoded_representation(model, torch.cat((pca_nmf_torch,covar_torch),axis=1))

adata.obsm["X_pca_nmf_ortho_structed"] = deepcopy(latents)
sc.pp.neighbors(adata, n_neighbors=10,
                n_pcs=adata.obsm["X_pca_nmf_ortho_structed"].shape[1],
                use_rep="X_pca_nmf_ortho_structed")
sc.tl.umap(adata, n_components=n_dims)
adata.obsm["X_pca_nmf_ortho_structed_umap"] = deepcopy(adata.obsm["X_umap"])
plt.scatter(adata.obsm["X_pca_nmf_ortho_structed_umap"][:, 0],
            adata.obsm["X_pca_nmf_ortho_structed_umap"][:, 1],
            c=c_data, cmap='inferno', s=1)
plt.set_title("PCA_NMF_corrected_umap")
plt.show()


sc.pl.scatter(adata, basis="pca_nmf_umap",
            color=["PECAM1","FOXP3","ICOS","GATA3","RUNX2","library"])


