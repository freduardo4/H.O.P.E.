import './otel-setup';
import { NestFactory, HttpAdapterHost } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe, Logger } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { AllExceptionsFilter } from './common/filters/all-exceptions.filter';
import { Logger as PinoLogger } from 'nestjs-pino';

async function bootstrap() {
    const app = await NestFactory.create(AppModule, { bufferLogs: true });

    // Use Pino for logging
    app.useLogger(app.get(PinoLogger));

    // Global Validation Pipe
    app.useGlobalPipes(new ValidationPipe({
        whitelist: true,
        transform: true,
        forbidNonWhitelisted: true,
    }));

    // Global Exception Filter
    const httpAdapter = app.get(HttpAdapterHost);
    app.useGlobalFilters(new AllExceptionsFilter(httpAdapter));

    // Swagger API Documentation
    const config = new DocumentBuilder()
        .setTitle('HOPE API')
        .setDescription('High-Output Performance Engineering API')
        .setVersion('1.0')
        .addBearerAuth()
        .build();
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api', app, document);

    const port = process.env.PORT || 3000;
    const logger = new Logger('Bootstrap');
    await app.listen(port);
    logger.log(`Application is running on: ${await app.getUrl()}`);
    logger.log(`Swagger Docs available at: ${await app.getUrl()}/api`);
}
bootstrap();
