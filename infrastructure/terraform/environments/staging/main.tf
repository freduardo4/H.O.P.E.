module "storage" {
  source      = "../../modules/storage"
  bucket_name = "hope-artifacts-staging"
  environment = "staging"
}

module "compute" {
  source    = "../../modules/compute"
  repo_name = "hope-ml-training-staging"
  tags = {
    Environment = "staging"
    Project     = "HOPE"
  }
}

output "artifacts_bucket" {
  value = module.storage.bucket_id
}

output "ecr_url" {
  value = module.compute.repository_url
}
