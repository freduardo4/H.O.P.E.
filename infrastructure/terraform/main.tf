terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for ML Artifacts and Models
resource "aws_s3_bucket" "hope_artifacts" {
  bucket = "hope-pipeline-artifacts-${var.environment}"

  tags = {
    Name        = "HOPE Artifacts"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "hope_artifacts_ver" {
  bucket = aws_s3_bucket.hope_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Placeholder for ECR Repository
resource "aws_ecr_repository" "hope_ml_repo" {
  name                 = "hope-ml-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
