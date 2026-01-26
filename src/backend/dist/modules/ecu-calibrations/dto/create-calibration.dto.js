"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CreateCalibrationDto = exports.ModificationDto = exports.MapDataDto = void 0;
const class_validator_1 = require("class-validator");
const class_transformer_1 = require("class-transformer");
const ecu_calibration_entity_1 = require("../entities/ecu-calibration.entity");
class MapDataDto {
}
exports.MapDataDto = MapDataDto;
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], MapDataDto.prototype, "name", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], MapDataDto.prototype, "address", void 0);
__decorate([
    (0, class_validator_1.IsArray)(),
    (0, class_validator_1.IsNumber)({}, { each: true }),
    __metadata("design:type", Array)
], MapDataDto.prototype, "originalValues", void 0);
__decorate([
    (0, class_validator_1.IsArray)(),
    (0, class_validator_1.IsNumber)({}, { each: true }),
    __metadata("design:type", Array)
], MapDataDto.prototype, "modifiedValues", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], MapDataDto.prototype, "unit", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], MapDataDto.prototype, "description", void 0);
class ModificationDto {
}
exports.ModificationDto = ModificationDto;
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "type", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "parameter", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "originalValue", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "newValue", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "timestamp", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], ModificationDto.prototype, "technicianId", void 0);
class CreateCalibrationDto {
}
exports.CreateCalibrationDto = CreateCalibrationDto;
__decorate([
    (0, class_validator_1.IsUUID)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "vehicleId", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "name", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "description", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "version", void 0);
__decorate([
    (0, class_validator_1.IsUUID)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "parentVersionId", void 0);
__decorate([
    (0, class_validator_1.IsEnum)(ecu_calibration_entity_1.CalibrationStatus),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "status", void 0);
__decorate([
    (0, class_validator_1.IsEnum)(ecu_calibration_entity_1.CalibrationProtocol),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "protocol", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "ecuPartNumber", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "ecuSoftwareNumber", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "ecuHardwareNumber", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "originalChecksum", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "modifiedChecksum", void 0);
__decorate([
    (0, class_validator_1.IsNumber)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", Number)
], CreateCalibrationDto.prototype, "fileSize", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "s3Key", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "originalS3Key", void 0);
__decorate([
    (0, class_validator_1.IsArray)(),
    (0, class_validator_1.ValidateNested)({ each: true }),
    (0, class_transformer_1.Type)(() => MapDataDto),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", Array)
], CreateCalibrationDto.prototype, "mapData", void 0);
__decorate([
    (0, class_validator_1.IsArray)(),
    (0, class_validator_1.ValidateNested)({ each: true }),
    (0, class_transformer_1.Type)(() => ModificationDto),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", Array)
], CreateCalibrationDto.prototype, "modifications", void 0);
__decorate([
    (0, class_validator_1.IsString)(),
    (0, class_validator_1.IsOptional)(),
    __metadata("design:type", String)
], CreateCalibrationDto.prototype, "notes", void 0);
//# sourceMappingURL=create-calibration.dto.js.map