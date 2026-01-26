"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiagnosticsModule = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const diagnostics_controller_1 = require("./diagnostics.controller");
const diagnostics_service_1 = require("./diagnostics.service");
const diagnostic_session_entity_1 = require("./entities/diagnostic-session.entity");
const obd2_reading_entity_1 = require("./entities/obd2-reading.entity");
const auth_1 = require("../auth");
let DiagnosticsModule = class DiagnosticsModule {
};
exports.DiagnosticsModule = DiagnosticsModule;
exports.DiagnosticsModule = DiagnosticsModule = __decorate([
    (0, common_1.Module)({
        imports: [
            typeorm_1.TypeOrmModule.forFeature([diagnostic_session_entity_1.DiagnosticSession, obd2_reading_entity_1.OBD2Reading]),
            auth_1.AuthModule,
        ],
        controllers: [diagnostics_controller_1.DiagnosticsController],
        providers: [diagnostics_service_1.DiagnosticsService],
        exports: [diagnostics_service_1.DiagnosticsService],
    })
], DiagnosticsModule);
//# sourceMappingURL=diagnostics.module.js.map