"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AppModule = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const health_1 = require("./health");
const auth_1 = require("./modules/auth");
const vehicles_1 = require("./modules/vehicles");
const diagnostics_1 = require("./modules/diagnostics");
const ecu_calibrations_1 = require("./modules/ecu-calibrations");
const reports_1 = require("./modules/reports");
const customers_1 = require("./modules/customers");
const config_1 = require("./config");
let AppModule = class AppModule {
};
exports.AppModule = AppModule;
exports.AppModule = AppModule = __decorate([
    (0, common_1.Module)({
        imports: [
            typeorm_1.TypeOrmModule.forRoot((0, config_1.getDatabaseConfig)()),
            health_1.HealthModule,
            auth_1.AuthModule,
            vehicles_1.VehiclesModule,
            diagnostics_1.DiagnosticsModule,
            ecu_calibrations_1.ECUCalibrationsModule,
            reports_1.ReportsModule,
            customers_1.CustomersModule,
        ],
        controllers: [],
        providers: [],
    })
], AppModule);
//# sourceMappingURL=app.module.js.map