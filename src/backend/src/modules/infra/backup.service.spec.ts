import { Test, TestingModule } from '@nestjs/testing';
import { BackupService } from './backup.service';
import { Logger } from '@nestjs/common';
import * as fs from 'fs';

// Mock S3Client
jest.mock('@aws-sdk/client-s3', () => {
    return {
        S3Client: jest.fn().mockImplementation(() => ({
            send: jest.fn().mockResolvedValue({}),
        })),
        PutObjectCommand: jest.fn(),
    };
});

describe('BackupService', () => {
    let service: BackupService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [BackupService],
        }).compile();

        service = module.get<BackupService>(BackupService);
    });

    it('should be defined', () => {
        expect(service).toBeDefined();
    });

    describe('performBackup', () => {
        it('should execute backup process without crashing', async () => {
            const loggerSpy = jest.spyOn(Logger.prototype, 'log');

            // Mock fs.existsSync to return true so it doesn't try to create dir
            jest.spyOn(fs, 'existsSync').mockReturnValue(true);

            await service.performBackup();

            expect(loggerSpy).toHaveBeenCalledWith(expect.stringContaining('Executing pg_dump'));
            expect(loggerSpy).toHaveBeenCalledWith(expect.stringContaining('Backup successfully uploaded to S3'));
        });

        it('should create temp directory if it does not exist', async () => {
            jest.spyOn(fs, 'existsSync').mockReturnValue(false);
            const mkdirSpy = jest.spyOn(fs, 'mkdirSync').mockImplementation();

            await service.performBackup();

            expect(mkdirSpy).toHaveBeenCalled();
        });
    });

    describe('handleDailyBackup', () => {
        it('should call performBackup', async () => {
            const performBackupSpy = jest.spyOn(service, 'performBackup').mockResolvedValue(undefined);
            await service.handleDailyBackup();
            expect(performBackupSpy).toHaveBeenCalled();
        });

        it('should log error if performBackup fails', async () => {
            const errorSpy = jest.spyOn(Logger.prototype, 'error');
            jest.spyOn(service, 'performBackup').mockRejectedValue(new Error('Backup failed'));

            await service.handleDailyBackup();

            expect(errorSpy).toHaveBeenCalledWith('Daily backup failed', expect.any(String));
        });
    });
});
