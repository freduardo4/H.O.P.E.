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
exports.OBD2Reading = void 0;
const typeorm_1 = require("typeorm");
let OBD2Reading = class OBD2Reading {
};
exports.OBD2Reading = OBD2Reading;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], OBD2Reading.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], OBD2Reading.prototype, "sessionId", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp' }),
    __metadata("design:type", Date)
], OBD2Reading.prototype, "timestamp", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], OBD2Reading.prototype, "pid", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], OBD2Reading.prototype, "name", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 12, scale: 4 }),
    __metadata("design:type", Number)
], OBD2Reading.prototype, "value", void 0);
__decorate([
    (0, typeorm_1.Column)(),
    __metadata("design:type", String)
], OBD2Reading.prototype, "unit", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], OBD2Reading.prototype, "rawResponse", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], OBD2Reading.prototype, "createdAt", void 0);
exports.OBD2Reading = OBD2Reading = __decorate([
    (0, typeorm_1.Entity)('obd2_readings'),
    (0, typeorm_1.Index)(['sessionId', 'timestamp']),
    (0, typeorm_1.Index)(['sessionId', 'pid'])
], OBD2Reading);
//# sourceMappingURL=obd2-reading.entity.js.map