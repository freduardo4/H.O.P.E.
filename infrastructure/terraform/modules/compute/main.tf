variable "repo_name" {
  description = "Name of the ECR repository"
  type        = string
}

variable "tags" {
  description = "Tags for the repository"
  type        = map(string)
  default     = {}
}

resource "aws_ecr_repository" "this" {
  name                 = var.repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = var.tags
}

output "repository_url" {
  value = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  value = aws_ecr_repository.this.arn
}
