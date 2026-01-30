// import { NodeSDK } from '@opentelemetry/sdk-node';
// import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
// import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';

// import { Resource } from '@opentelemetry/resources';

// const traceExporter = new OTLPTraceExporter({
//     url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4317',
// });

// export const otraceSDK = new NodeSDK({
//     // @ts-ignore
//     resource: new Resource({
//         'service.name': 'hope-backend',
//         'deployment.environment': process.env.NODE_ENV || 'development',
//     }),
//     traceExporter,
//     instrumentations: [getNodeAutoInstrumentations()],
// });

// // Start the SDK
// if (process.env.ENABLE_OTEL === 'true') {
//     otraceSDK.start();
//     console.log('OpenTelemetry initialized');
// }

// // Graceful shutdown
// process.on('SIGTERM', () => {
//     otraceSDK.shutdown()
//         .then(() => console.log('Tracing terminated'))
//         .catch((error) => console.log('Error terminating tracing', error))
//         .finally(() => process.exit(0));
// });
export const otraceSDK = {};

