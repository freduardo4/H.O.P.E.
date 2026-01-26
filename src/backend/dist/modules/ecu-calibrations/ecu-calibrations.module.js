"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ECUCalibrationsModule = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const ecu_calibrations_controller_1 = require("./ecu-calibrations.controller");
const ecu_calibrations_service_1 = require("./ecu-calibrations.service");
const ecu_calibration_entity_1 = require("./entities/ecu-calibration.entity");
const auth_1 = require("../auth");
let ECUCalibrationsModule = class ECUCalibrationsModule {
};
exports.ECUCalibrationsModule = ECUCalibrationsModule;
exports.ECUCalibrationsModule = ECUCalibrationsModule = __decorate([
    (0, common_1.Module)({
        imports: [typeorm_1.TypeOrmModule.forFeature([ecu_calibration_entity_1.ECUCalibration]), auth_1.AuthModule],
        controllers: [ecu_calibrations_controller_1.ECUCalibrationsController],
        providers: [ecu_calibrations_service_1.ECUCalibrationsService],
        exports: [ecu_calibrations_service_1.ECUCalibrationsService],
    })
], ECUCalibrationsModule);
//# sourceMappingURL=ecu-calibrations.module.js.map