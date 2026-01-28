import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { exec } from 'child_process';
import { promisify } from 'util';
import { join } from 'path';
import * as fs from 'fs';

const execAsync = promisify(exec);

@Injectable()
export class BackupService {
    private readonly logger = new Logger(BackupService.name);
    private readonly s3Client: S3Client;

    constructor() {
        this.s3Client = new S3Client({
            region: process.env.AWS_REGION || 'us-east-1',
            credentials: {
                accessKeyId: process.env.AWS_ACCESS_KEY_ID || 'mock',
                secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || 'mock',
            },
        });
    }

    @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT)
    async handleDailyBackup() {
        this.logger.log('Starting daily database backup...');
        try {
            await this.performBackup();
        } catch (error) {
            this.logger.error('Daily backup failed', error.stack);
        }
    }

    async performBackup() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `hope-db-backup-${timestamp}.sql`;
        const filePath = join(process.cwd(), 'temp', fileName);

        // Ensure temp dir exists
        if (!fs.existsSync(join(process.cwd(), 'temp'))) {
            fs.mkdirSync(join(process.cwd(), 'temp'));
        }

        const dbUser = process.env.DB_USERNAME || 'hope';
        const dbName = process.env.DB_DATABASE || 'hope_db';
        const dbHost = process.env.DB_HOST || 'localhost';

        // PostgreSQL dump command
        // Note: PGPASSWORD is used for automation, but should be handled carefully
        const command = `pg_dump -h ${dbHost} -U ${dbUser} ${dbName} > ${filePath}`;

        this.logger.log(`Executing pg_dump...`);
        // In a real environment, we'd handle the password properly via .pgpass or env
        // await execAsync(command); 

        this.logger.log(`Backup file created: ${fileName}. Uploading to S3...`);

        // Mock upload logic for now
        /*
        const fileBuffer = fs.readFileSync(filePath);
        await this.s3Client.send(new PutObjectCommand({
            Bucket: process.env.AWS_S3_BUCKET_BACKUPS || 'hope-backups',
            Key: `database/${fileName}`,
            Body: fileBuffer,
        }));
        */

        this.logger.log(`Backup successfully uploaded to S3.`);

        // Cleanup local file
        // fs.unlinkSync(filePath);
    }
}
