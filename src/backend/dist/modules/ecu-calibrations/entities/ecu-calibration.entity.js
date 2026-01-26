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
exports.ECUCalibration = exports.FileFormat = exports.CalibrationType = void 0;
const typeorm_1 = require("typeorm");
var CalibrationType;
(function (CalibrationType) {
    CalibrationType["STOCK"] = "stock";
    CalibrationType["STAGE_1"] = "stage1";
    CalibrationType["STAGE_2"] = "stage2";
    CalibrationType["STAGE_3"] = "stage3";
    CalibrationType["CUSTOM"] = "custom";
})(CalibrationType || (exports.CalibrationType = CalibrationType = {}));
var FileFormat;
(function (FileFormat) {
    FileFormat["BIN"] = "bin";
    FileFormat["HEX"] = "hex";
    FileFormat["S19"] = "s19";
    FileFormat["UNKNOWN"] = "unknown";
})(FileFormat || (exports.FileFormat = FileFormat = {}));
let ECUCalibration = class ECUCalibration {
};
exports.ECUCalibration = ECUCalibration;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], ECUCalibration.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "tenantId", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "vehicleId", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "customerId", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "fileName", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "s3Key", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "s3Bucket", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'bigint' }),
    __metadata("design:type", Number)
], ECUCalibration.prototype, "fileSize", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'enum', enum: FileFormat, default: FileFormat.UNKNOWN }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "fileFormat", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'enum', enum: CalibrationType }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "calibrationType", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], ECUCalibration.prototype, "checksum", void 0);
__decorate([
    (0, typeorm_1.Column)({ default: 1 }),
    __metadata("design:type", Number)
], ECUCalibration.prototype, "version", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "previousVersionId", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "ecuType", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "ecuSoftwareVersion", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "notes", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', nullable: true }),
    __metadata("design:type", Object)
], ECUCalibration.prototype, "metadata", void 0);
__decorate([
    (0, typeorm_1.Column)({ default: true }),
    __metadata("design:type", Boolean)
], ECUCalibration.prototype, "isActive", void 0);
__decorate([
    (0, typeorm_1.Column)({ nullable: true }),
    __metadata("design:type", String)
], ECUCalibration.prototype, "uploadedBy", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], ECUCalibration.prototype, "createdAt", void 0);
__decorate([
    (0, typeorm_1.UpdateDateColumn)(),
    __metadata("design:type", Date)
], ECUCalibration.prototype, "updatedAt", void 0);
exports.ECUCalibration = ECUCalibration = __decorate([
    (0, typeorm_1.Entity)('ecu_calibrations')
], ECUCalibration);
//# sourceMappingURL=ecu-calibration.entity.js.map