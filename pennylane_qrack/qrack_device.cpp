#include <regex>
#include <stdexcept>
#include <string>
#include <QuantumDevice.hpp>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "qrack/qfactory.hpp"

std::string trim(std::string s)
{
    // Cut leading, trailing, and extra spaces
    // (See https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string#answer-1798170)
    return std::regex_replace(s, std::regex("^ +| +$|( ) +"), "$1");
}

struct QrackObservable {
    std::vector<Qrack::Pauli> obs;
    std::vector<bitLenInt> wires;
    QrackObservable()
    {
        // Intentionally left blank
    }
    QrackObservable(std::vector<Qrack::Pauli> o, std::vector<bitLenInt> w)
        : obs(o)
        , wires(w)
    {
        // Intentionally left blank
    }
};

struct QrackDevice final : public Catalyst::Runtime::QuantumDevice {
    bool tapeRecording;
    size_t shots;
    Qrack::QInterfacePtr qsim;
    std::vector<QrackObservable> obs_cache;

    // static constants for RESULT values
    static constexpr bool QRACK_RESULT_TRUE_CONST = true;
    static constexpr bool QRACK_RESULT_FALSE_CONST = false;

    inline auto getDeviceWires(const std::vector<QubitIdType> &wires) -> std::vector<bitLenInt>
    {
        std::vector<bitLenInt> res;
        res.reserve(wires.size());
        std::transform(wires.begin(), wires.end(), std::back_inserter(res), [](auto w) { return (bitLenInt)w; });
        return res;
    }

    inline auto wiresToMask(const std::vector<bitLenInt> &wires) -> bitCapInt
    {
        bitCapInt mask = Qrack::ZERO_BCI;
        for (const bitLenInt& target : wires) {
            mask = mask | Qrack::pow2(target);
        }

        return mask;
    }

    void applyNamedOperation(const std::string &name, const std::vector<bitLenInt> &wires,
                             const bool& inverse, const std::vector<double> &params)
    {
        if (name == "PauliX") {
            // Self-adjoint, so ignore "inverse"
            if (wires.size() > 1U) {
                qsim->XMask(wiresToMask(wires));
            } else {
                qsim->X(wires[0U]);
            }
        } else if (name == "PauliY") {
            // Self-adjoint, so ignore "inverse"
            if (wires.size() > 1U) {
                qsim->YMask(wiresToMask(wires));
            } else {
                qsim->Y(wires[0U]);
            }
        } else if (name == "PauliZ") {
            // Self-adjoint, so ignore "inverse"
            if (wires.size() > 1U) {
                qsim->ZMask(wiresToMask(wires));
            } else {
                qsim->Z(wires[0U]);
            }
        } else if (name == "SX") {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->ISqrtX(target);
                } else {
                    qsim->SqrtX(target);
                }
            }
        } else if (name == "MultiRZ") {
            for (const bitLenInt& target : wires) {
                qsim->RZ(inverse ? -params[0U] : params[0U], target);
            }
        } else if (name == "Hadamard") {
            for (const bitLenInt& target : wires) {
                qsim->H(target);
            }
        } else if (name == "S") {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->IS(target);
                } else {
                    qsim->S(target);
                }
            }
        } else if (name == "T") {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->IT(target);
                } else {
                    qsim->T(target);
                }
            }
        } else if (name == "SWAP") {
            if (wires.size() != 2U) {
                throw std::invalid_argument("SWAP must have exactly two target qubits!");
            }
            qsim->Swap(wires[0U], wires[1U]);
        } else if (name == "ISWAP") {
            if (wires.size() != 2U) {
                throw std::invalid_argument("ISWAP must have exactly two target qubits!");
            }
            qsim->ISwap(wires[0U], wires[1U]);
        } else if (name == "PSWAP") {
            if (wires.size() != 2U) {
                throw std::invalid_argument("PSWAP must have exactly two target qubits!");
            }
            const std::vector<bitLenInt> c { wires[0U] };
            qsim->CU(c, wires[1U], ZERO_R1, ZERO_R1, inverse ? -params[0U] : params[0U]);
            qsim->Swap(wires[0U], wires[1U]);
            qsim->CU(c, wires[1U], ZERO_R1, ZERO_R1, inverse ? -params[0U] : params[0U]);
        } else if (name == "PhaseShift") {
            const Qrack::real1 cosine = (Qrack::real1)cos(inverse ? -params[0U] : params[0U]);
            const Qrack::real1 sine = (Qrack::real1)sin(inverse ? -params[0U] : params[0U]);
            const Qrack::complex bottomRight(cosine, sine);
            for (const bitLenInt& target : wires) {
                qsim->Phase(Qrack::ONE_CMPLX, bottomRight, target);
            }
        } else if (name == "PhaseShift") {
            const Qrack::real1 cosine = (Qrack::real1)cos(inverse ? -params[0U] : params[0U]);
            const Qrack::real1 sine = (Qrack::real1)sin(inverse ? -params[0U] : params[0U]);
            const Qrack::complex bottomRight(cosine, sine);
            for (const bitLenInt& target : wires) {
                qsim->Phase(Qrack::ONE_CMPLX, bottomRight, target);
            }
        } else if ((name == "ControlledPhaseShift") || (name == "CPhase")) {
            const Qrack::real1 cosine = (Qrack::real1)cos(inverse ? -params[0U] : params[0U]);
            const Qrack::real1 sine = (Qrack::real1)sin(inverse ? -params[0U] : params[0U]);
            const Qrack::complex bottomRight(cosine, sine);
            qsim->MCPhase(std::vector<bitLenInt>(wires.begin(), wires.end() - 1U), Qrack::ONE_CMPLX, bottomRight, wires.back());
        } else if (name == "RX") {
            for (const bitLenInt& target : wires) {
                qsim->RX(inverse ? -params[0U] : params[0U], target);
            }
        } else if (name == "RY") {
            for (const bitLenInt& target : wires) {
                qsim->RY(inverse ? -params[0U] : params[0U], target);
            }
        } else if (name == "RZ") {
            for (const bitLenInt& target : wires) {
                qsim->RZ(inverse ? -params[0U] : params[0U], target);
            }
        } else if ((name == "Rot") || (name == "U3")) {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->U(target, -params[0U], -params[1U], -params[2U]);
                } else {
                    qsim->U(target, params[0U], params[1U], params[2U]);
                }
            }
        } else if (name == "U2") {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->U(target, -Qrack::PI_R1 / 2, -params[0U], -params[1U]);
                } else {
                    qsim->U(target, Qrack::PI_R1 / 2, params[0U], params[1U]);
                }
            }
        } else if (name == "U1") {
            for (const bitLenInt& target : wires) {
                if (inverse) {
                    qsim->U(target, ZERO_R1, ZERO_R1, -params[0U]);
                } else {
                    qsim->U(target, ZERO_R1, ZERO_R1, params[0U]);
                }
            }
        } else if (name == "QFT") {
            if (inverse) {
                qsim->IQFTR(wires);
            } else {
                qsim->QFTR(wires);
            }
        } else if (name != "Identity") {
            throw std::domain_error("Unrecognized gate name: " + name);
        }
    }

    void applyNamedOperation(const std::string &name, const std::vector<bitLenInt> &control_wires,
                             const std::vector<bool> &control_values,
                             const std::vector<bitLenInt> &wires, const bool& inverse,
                             const std::vector<double> &params)
    {
        bitCapInt controlPerm = 0U;
        for (size_t i = 0U; i < control_values.size(); ++i) {
            controlPerm = controlPerm << 1U;
            if (control_values[i]) {
                controlPerm = controlPerm | 1U;
            }
        }

        QRACK_CONST Qrack::complex SQRT1_2_CMPLX(Qrack::SQRT1_2_R1, ZERO_R1);
        QRACK_CONST Qrack::complex NEG_SQRT1_2_CMPLX(-Qrack::SQRT1_2_R1, ZERO_R1);
        QRACK_CONST Qrack::complex SQRTI_2_CMPLX(ZERO_R1, Qrack::SQRT1_2_R1);
        QRACK_CONST Qrack::complex NEG_SQRTI_2_CMPLX(ZERO_R1, -Qrack::SQRT1_2_R1);
        const Qrack::complex QBRTI_2_CMPLX(ZERO_R1, sqrt(Qrack::SQRT1_2_R1));
        const Qrack::complex NEG_QBRTI_2_CMPLX(ZERO_R1, sqrt(-Qrack::SQRT1_2_R1));
        QRACK_CONST Qrack::complex ONE_PLUS_I_DIV_2 = Qrack::complex((Qrack::real1)(ONE_R1 / 2), (Qrack::real1)(ONE_R1 / 2));
        QRACK_CONST Qrack::complex ONE_MINUS_I_DIV_2 = Qrack::complex((Qrack::real1)(ONE_R1 / 2), (Qrack::real1)(-ONE_R1 / 2));

        QRACK_CONST Qrack::complex pauliX[4U] = { Qrack::ZERO_CMPLX, Qrack::ONE_CMPLX, Qrack::ONE_CMPLX, Qrack::ZERO_CMPLX };
        QRACK_CONST Qrack::complex pauliY[4U] = { Qrack::ZERO_CMPLX, -Qrack::I_CMPLX, Qrack::I_CMPLX, Qrack::ZERO_CMPLX };
        QRACK_CONST Qrack::complex pauliZ[4U] = { Qrack::ONE_CMPLX, Qrack::ZERO_CMPLX, Qrack::ZERO_CMPLX, -Qrack::ONE_CMPLX };
        QRACK_CONST Qrack::complex sqrtX[4U]{ ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2 };
        QRACK_CONST Qrack::complex iSqrtX[4U]{ ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2 };
        QRACK_CONST Qrack::complex hadamard[4U]{ SQRT1_2_CMPLX, SQRT1_2_CMPLX, SQRT1_2_CMPLX, NEG_SQRT1_2_CMPLX };

        if ((name == "PauliX") || (name == "CNOT") || (name == "Toffoli") || (name == "MultiControlledX")) {
            // Self-adjoint, so ignore "inverse"
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, pauliX, target, controlPerm);
            }
        } else if ((name == "PauliY") || (name == "CY")) {
            // Self-adjoint, so ignore "inverse"
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, pauliY, target, controlPerm);
            }
        } else if ((name == "PauliZ") || (name == "CZ")) {
            // Self-adjoint, so ignore "inverse"
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, pauliZ, target, controlPerm);
            }
        } else if (name == "SX") {
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, inverse ? iSqrtX : sqrtX, target, controlPerm);
            }
        } else if (name == "MultiRZ") {
            const Qrack::real1 cosine = (Qrack::real1)cos((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::real1 sine = (Qrack::real1)sin((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::complex topLeft(cosine, -sine);
            const Qrack::complex bottomRight(cosine, sine);
            for (const bitLenInt& target : wires) {
                qsim->UCPhase(control_wires, topLeft, bottomRight, target, controlPerm);
            }
        } else if (name == "Hadamard") {
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, hadamard, target, controlPerm);
            }
        } else if (name == "S") {
            for (const bitLenInt& target : wires) {
                qsim->UCPhase(control_wires, Qrack::ONE_CMPLX, inverse ? NEG_SQRTI_2_CMPLX : SQRTI_2_CMPLX, target, controlPerm);
            }
        } else if (name == "T") {
            for (const bitLenInt& target : wires) {
                qsim->UCPhase(control_wires, Qrack::ONE_CMPLX, inverse ? NEG_QBRTI_2_CMPLX : QBRTI_2_CMPLX, target, controlPerm);
            }
        } else if ((name == "SWAP") || (name == "CSWAP")) {
            if (wires.size() != 2U) {
                throw std::invalid_argument("SWAP and CSWAP must have exactly two target qubits!");
            }
            for (bitLenInt i = 0U; i < control_wires.size(); ++i) {
                if (!control_values[i]) {
                    qsim->X(control_wires[i]);
                }
            }
            qsim->CSwap(control_wires, wires[0U], wires[1U]);
            for (bitLenInt i = 0U; i < control_wires.size(); ++i) {
                if (!control_values[i]) {
                    qsim->X(control_wires[i]);
                }
            }
        } else if (name == "ISWAP") {
            if (wires.size() != 2U) {
                throw std::invalid_argument("ISWAP must have exactly two target qubits!");
            }
            for (bitLenInt i = 0U; i < control_wires.size(); ++i) {
                if (!control_values[i]) {
                    qsim->X(control_wires[i]);
                }
            }
            std::vector<bitLenInt> mcp_wires(control_wires);
            mcp_wires.push_back(wires[0U]);
            qsim->MCPhase(mcp_wires, Qrack::I_CMPLX, Qrack::ONE_CMPLX, wires[1U]);
            qsim->CSwap(control_wires, wires[0U], wires[1U]);
            qsim->MCPhase(mcp_wires, Qrack::I_CMPLX, Qrack::ONE_CMPLX, wires[1U]);
            for (bitLenInt i = 0U; i < control_wires.size(); ++i) {
                if (!control_values[i]) {
                    qsim->X(control_wires[i]);
                }
            }
        } else if ((name == "PhaseShift") || (name == "ControlledPhaseShift") || (name == "CPhase")) {
            const Qrack::real1 cosine = (Qrack::real1)cos(inverse ? -params[0U] : params[0U]);
            const Qrack::real1 sine = (Qrack::real1)sin(inverse ? -params[0U] : params[0U]);
            const Qrack::complex bottomRight(cosine, sine);
            for (const bitLenInt& target : wires) {
                qsim->UCPhase(control_wires, Qrack::ONE_CMPLX, bottomRight, target, controlPerm);
            }
        } else if (name == "PSWAP") {
            std::vector<bitLenInt> c(control_wires);
            c.push_back(wires[0U]);
            qsim->CU(c, wires[1U], ZERO_R1, ZERO_R1, inverse ? -params[0U] : params[0U]);
            qsim->CSwap(control_wires, wires[0U], wires[1U]);
            qsim->CU(c, wires[1U], ZERO_R1, ZERO_R1, inverse ? -params[0U] : params[0U]);
        } else if ((name == "RX") || (name == "CRX")) {
            const Qrack::real1 cosine = (Qrack::real1)cos((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::real1 sine = (Qrack::real1)sin((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::complex mtrx[4U] = {
                Qrack::complex(cosine, ZERO_R1), Qrack::complex(ZERO_R1, -sine),
                Qrack::complex(ZERO_R1, -sine), Qrack::complex(cosine, ZERO_R1)
            };
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, mtrx, target, controlPerm);
            }
        } else if ((name == "RY") || (name == "CRY")) {
            const Qrack::real1 cosine = (Qrack::real1)cos((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::real1 sine = (Qrack::real1)sin((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::complex mtrx[4U] = {
                Qrack::complex(cosine, ZERO_R1), Qrack::complex(-sine, ZERO_R1),
                Qrack::complex(sine, ZERO_R1), Qrack::complex(cosine, ZERO_R1)
            };
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, mtrx, target, controlPerm);
            }
        } else if ((name == "RZ") || (name == "CRZ")) {
            const Qrack::real1 cosine = (Qrack::real1)cos((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::real1 sine = (Qrack::real1)sin((inverse ? -params[0U] : params[0U]) / 2);
            const Qrack::complex topLeft(cosine, -sine);
            const Qrack::complex bottomRight(cosine, sine);
            for (const bitLenInt& target : wires) {
                qsim->UCPhase(control_wires, topLeft, bottomRight, target, controlPerm);
            }
        } else if ((name == "Rot") || (name == "U3") || (name == "CRot")) {
            const Qrack::real1 theta = inverse ? -params[0U] : params[0U];
            const Qrack::real1 phi = inverse ? -params[1U] : params[1U];
            const Qrack::real1 lambda = inverse ? -params[2U] : params[2U];
            const Qrack::real1 cos0 = (Qrack::real1)cos(theta / 2);
            const Qrack::real1 sin0 = (Qrack::real1)sin(theta / 2);
            const Qrack::complex mtrx[4]{
                Qrack::complex(cos0, ZERO_R1), sin0 * Qrack::complex((Qrack::real1)(-cos(lambda)),
                (Qrack::real1)(-sin(lambda))),
                sin0 * Qrack::complex((Qrack::real1)cos(phi), (Qrack::real1)sin(phi)),
                cos0 * Qrack::complex((Qrack::real1)cos(phi + lambda), (Qrack::real1)sin(phi + lambda))
            };
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, mtrx, target, controlPerm);
            }
        } else if (name == "U2") {
            const Qrack::real1 theta = (inverse ? -Qrack::PI_R1 : Qrack::PI_R1) / 2;
            const Qrack::real1 phi = inverse ? -params[0U] : params[0U];
            const Qrack::real1 lambda = inverse ? -params[1U] : params[1U];
            const Qrack::real1 cos0 = (Qrack::real1)cos(theta / 2);
            const Qrack::real1 sin0 = (Qrack::real1)sin(theta / 2);
            const Qrack::complex mtrx[4]{
                Qrack::complex(cos0, ZERO_R1), sin0 * Qrack::complex((Qrack::real1)(-cos(lambda)),
                (Qrack::real1)(-sin(lambda))),
                sin0 * Qrack::complex((Qrack::real1)cos(phi), (Qrack::real1)sin(phi)),
                cos0 * Qrack::complex((Qrack::real1)cos(phi + lambda), (Qrack::real1)sin(phi + lambda))
            };
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, mtrx, target, controlPerm);
            }
        } else if (name == "U1") {
            const Qrack::real1 theta = ZERO_R1;
            const Qrack::real1 phi = ZERO_R1;
            const Qrack::real1 lambda = inverse ? -params[0U] : params[0U];
            const Qrack::real1 cos0 = (Qrack::real1)cos(theta / 2);
            const Qrack::real1 sin0 = (Qrack::real1)sin(theta / 2);
            const Qrack::complex mtrx[4]{
                Qrack::complex(cos0, ZERO_R1), sin0 * Qrack::complex((Qrack::real1)(-cos(lambda)),
                (Qrack::real1)(-sin(lambda))),
                sin0 * Qrack::complex((Qrack::real1)cos(phi), (Qrack::real1)sin(phi)),
                cos0 * Qrack::complex((Qrack::real1)cos(phi + lambda), (Qrack::real1)sin(phi + lambda))
            };
            for (const bitLenInt& target : wires) {
                qsim->UCMtrx(control_wires, mtrx, target, controlPerm);
            }
        } else if (name != "Identity") {
            throw std::domain_error("Unrecognized gate name: " + name);
        }
    }

    QrackDevice([[maybe_unused]] std::string kwargs = "{}")
        : tapeRecording(false)
        , shots(1U)
        , qsim(nullptr)
    {
        // Cut leading '{' and trailing '}'
        kwargs.erase(0U, 1U);
        kwargs.erase(kwargs.size() - 1U);
        // Cut leading, trailing, and extra spaces
        kwargs = trim(kwargs);

        std::map<std::string, int> keyMap;
        keyMap["wires"] = 1;
        keyMap["shots"] = 2;
        keyMap["is_hybrid_stabilizer"] = 3;
        keyMap["is_tensor_network"] = 4;
        keyMap["is_schmidt_decomposed"] = 5;
        keyMap["is_schmidt_decomposition_parallel"] = 6;
        keyMap["is_qbdd"] = 7;
        keyMap["is_gpu"] = 8;
        keyMap["is_host_pointer"] = 9;

        bitLenInt wires = 24;
        bool is_hybrid_stabilizer = true;
        bool is_tensor_network = true;
        bool is_schmidt_decomposed = true;
        bool is_schmidt_decomposition_parallel = true;
        bool is_qbdd = false;
        bool is_gpu = true;
        bool is_host_pointer = false;

        size_t pos;
        while ((pos = kwargs.find(":")) != std::string::npos) {
            std::string key = kwargs.substr(0, pos);
            kwargs.erase(0, pos + 1U);
            // Leading and trailing quotes:
            key.erase(0U, 1U);
            key.erase(key.size() - 1U);

            if (key == "wires") {
                // Handle if integer
                pos = kwargs.find(",");
                bool isInt = true;
                for (size_t i = 0; i < pos; ++i) {
                    if ((kwargs[i] != ' ') && !isdigit(kwargs[i])) {
                        isInt = false;
                        break;
                    }
                }
                if (isInt) {
                    wires = stoi(trim(kwargs.substr(0, pos)));
                    kwargs.erase(0, pos + 1U);
                    continue;
                }

                // Handles if Wires object
                pos = kwargs.find("]>,");
                std::string value = kwargs.substr(0, pos);
                kwargs.erase(0, pos + 3U);
                size_t p = value.find("[");
                value.erase(0, p + 1U);
                wires = 0U;
                size_t q;
                while ((q = value.find(",")) != std::string::npos) {
                    const bitLenInt label = stoi(value.substr(0, q));
                    value.erase(0, q + 1U);
                    if (label > wires) {
                        wires = label;
                    }
                }
                ++wires;
                continue;
            }

            pos = kwargs.find(",");
            std::string value = trim(kwargs.substr(0, pos));
            kwargs.erase(0, pos + 1U);
            const bool val = (value == "True");
            switch (keyMap[key]) {
                case 2:
                    if (value != "None") {
                        shots = std::stoi(value);
                    }
                    break;
                case 3:
                    is_hybrid_stabilizer = val;
                    break;
                case 4:
                    is_tensor_network = val;
                    break;
                case 5:
                    is_schmidt_decomposed = val;
                    break;
                case 6:
                    is_schmidt_decomposition_parallel = val;
                    break;
                case 7:
                    is_qbdd = val;
                    break;
                case 8:
                    is_gpu =  val;
                    break;
                case 9:
                    is_host_pointer = val;
                    break;
                default:
                    break;
            }
        }

        // Construct backwards, then reverse:
        std::vector<Qrack::QInterfaceEngine> simulatorType;

        if (!is_gpu) {
            simulatorType.push_back(Qrack::QINTERFACE_CPU);
        }

        if (is_qbdd) {
            simulatorType.push_back(Qrack::QINTERFACE_BDT_HYBRID);
        }

        if (is_hybrid_stabilizer) {
            simulatorType.push_back(Qrack::QINTERFACE_STABILIZER_HYBRID);
        }

        if (is_schmidt_decomposed) {
            simulatorType.push_back(is_schmidt_decomposition_parallel ? Qrack::QINTERFACE_QUNIT_MULTI : Qrack::QINTERFACE_QUNIT);
        }

        if (is_tensor_network) {
            simulatorType.push_back(Qrack::QINTERFACE_TENSOR_NETWORK);
        }

        // (...then reverse:)
        std::reverse(simulatorType.begin(), simulatorType.end());

        if (!simulatorType.size()) {
            simulatorType.push_back(Qrack::QINTERFACE_CPU);
        }

        qsim = CreateQuantumInterface(simulatorType, wires, Qrack::ZERO_BCI, nullptr, Qrack::CMPLX_DEFAULT_ARG, false, true, is_host_pointer);
    }

    QrackDevice &operator=(const QuantumDevice &) = delete;
    QrackDevice(const QrackDevice &) = delete;
    QrackDevice(QrackDevice &&) = delete;
    QrackDevice &operator=(QuantumDevice &&) = delete;

    auto AllocateQubit() -> QubitIdType override {
        return qsim->Allocate(1U);
    }
    auto AllocateQubits(size_t num_qubits) -> std::vector<QubitIdType> override {
        std::vector<QubitIdType> ids(num_qubits);
        for (size_t i = 0U; i < num_qubits; ++i) {
            ids[i] = qsim->Allocate(1U);
        }
        return ids;
    }
    [[nodiscard]] auto Zero() const -> Result override { return const_cast<Result>(&QRACK_RESULT_FALSE_CONST); }
    [[nodiscard]] auto One() const -> Result override { return const_cast<Result>(&QRACK_RESULT_TRUE_CONST); }
    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires) -> ObsIdType override
    {
        Qrack::Pauli basis = Qrack::PauliI;
        switch (id) {
            case ObsId::PauliX:
                basis = Qrack::PauliX;
                break;
            case ObsId::PauliY:
                basis = Qrack::PauliY;
                break;
            case ObsId::PauliZ:
                basis = Qrack::PauliZ;
                break;
            default:
                break;
        }
        obs_cache.push_back(QrackObservable({ basis }, { wires[0U] }));

        return obs_cache.size() - 1U;
    }
    auto TensorObservable(const std::vector<ObsIdType> &obs) -> ObsIdType override
    {
        QrackObservable o;
        for (const ObsIdType& id : obs) {
            const QrackObservable& i = obs_cache[id];
            o.obs.insert(o.obs.end(), i.obs.begin(), i.obs.end());
            o.wires.insert(o.wires.end(), i.wires.begin(), i.wires.end());
        }
        obs_cache.push_back(o);

        return obs_cache.size() - 1U;
    }
    auto HamiltonianObservable(const std::vector<double> &coeffs, const std::vector<ObsIdType> &obs)
        -> ObsIdType override
    {
        return -1;
    }
    auto Measure(QubitIdType id, std::optional<int> postselect) -> Result override {
        bool *ret = (bool *)malloc(sizeof(bool));
        if (postselect.has_value()) {
            *ret = qsim->ForceM(id, postselect.value());
        } else {
            *ret = qsim->M(id);
        }
        return ret;
    }

    void ReleaseQubit(QubitIdType id) override
    {
        qsim->Dispose(id, 1U);
    }
    void ReleaseAllQubits() override
    {
        qsim->Dispose(0U, qsim->GetQubitCount());
    }
    [[nodiscard]] auto GetNumQubits() const -> size_t override
    {
        return qsim->GetQubitCount();
    }
    void SetDeviceShots(size_t s) override { shots = s; }
    [[nodiscard]] auto GetDeviceShots() const -> size_t override { return shots; }
    void StartTapeRecording() override { tapeRecording = true; }
    void StopTapeRecording() override { tapeRecording = false; }
    void PrintState() override
    {
        const size_t numQubits = qsim->GetQubitCount();
        const size_t maxQPower = (size_t)qsim->GetMaxQPower();
        const size_t maxLcv = maxQPower - 1U;
        size_t idx = 0U;
        std::cout << "*** State-Vector of Size " << maxQPower << " ***" << std::endl;
        std::cout << "[";
        std::unique_ptr<Qrack::complex> sv(new Qrack::complex[maxQPower]);
        qsim->GetQuantumState(sv.get());
        for (; idx < maxLcv; ++idx) {
            std::cout << sv.get()[idx] << ", ";
        }
        std::cout << sv.get()[idx] << "]" << std::endl;
    }
    void NamedOperation(const std::string &name, const std::vector<double> &params,
                        const std::vector<QubitIdType> &wires, bool inverse,
                        const std::vector<QubitIdType> &controlled_wires,
                        const std::vector<bool> &controlled_values) override
    {
        // Check the validity of number of qubits and parameters
        RT_FAIL_IF(controlled_wires.size() != controlled_values.size(), "Controlled wires/values size mismatch");

        // Convert wires to device wires
        auto &&dev_wires = getDeviceWires(wires);
        auto &&dev_controlled_wires = getDeviceWires(controlled_wires);
        std::vector<bool> dev_controlled_values(controlled_values);
        if (name == "MultiControlledX") {
            while (wires.size() > 1U) {
                dev_controlled_wires.push_back(dev_wires[0]);
                dev_controlled_values.push_back(true);
                dev_wires.erase(dev_wires.begin());
            }
        } else if (name == "Toffoli") {
            dev_controlled_wires.push_back(dev_wires[0]);
            dev_controlled_values.push_back(true);
            dev_wires.erase(dev_wires.begin());
            dev_controlled_wires.push_back(dev_wires[0]);
            dev_controlled_values.push_back(true);
            dev_wires.erase(dev_wires.begin());
        } else if ((name == "CNOT")
            || (name == "CY")
            || (name == "CZ")
            || (name == "CSWAP")
            || (name == "ControlledPhaseShift")
            || (name == "CRX")
            || (name == "CRY")
            || (name == "CRZ")
            || (name == "CRot"))
        {
            dev_controlled_wires.push_back(dev_wires[0]);
            dev_controlled_values.push_back(true);
            dev_wires.erase(dev_wires.begin());
        }

        // Update the state-vector
        if (controlled_wires.empty()) {
            applyNamedOperation(name, dev_wires, inverse, params);
        } else {
            applyNamedOperation(name, dev_controlled_wires, dev_controlled_values, dev_wires, inverse, params);
        }

        // TODO: This the Lightning implementation, which doesn't work for Qrack yet, below.

        // Update tape caching if required
        // if (tapeRecording) {
        //     this->cache_manager.addOperation(name, params, dev_wires, inverse, {}, dev_controlled_wires,
        //                                  controlled_values);
        // }
    }
    void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse,
                         const std::vector<QubitIdType> &controlled_wires,
                         const std::vector<bool> &controlled_values) override
    {
        RT_FAIL_IF(controlled_wires.size() != controlled_values.size(), "Controlled wires/values size mismatch");
        RT_FAIL_IF(wires.size() != 1U, "Matrix operation can only have one target qubit!");

        // Convert wires to device wires
        // with checking validity of wires
        auto &&dev_wires = getDeviceWires(wires);
        auto &&dev_controlled_wires = getDeviceWires(controlled_wires);
        const Qrack::complex mtrx[4U] = {
            (Qrack::complex)matrix[0U], (Qrack::complex)matrix[1U],
            (Qrack::complex)matrix[2U], (Qrack::complex)matrix[3U]
        };
        Qrack::complex inv[4U];
        Qrack::inv2x2(mtrx, inv);

        // Update the state-vector
        if (controlled_wires.empty()) {
            qsim->Mtrx(inverse ? inv : mtrx, wires[0U]);
        } else {
            bitCapInt controlPerm = 0U;
            for (size_t i = 0U; i < controlled_values.size(); ++i) {
                controlPerm = controlPerm << 1U;
                if (controlled_values[i]) {
                    controlPerm = controlPerm | 1U;
                }
            }
            qsim->UCMtrx(dev_controlled_wires, inverse ? inv : mtrx, wires[0U], controlPerm);
        }

        // TODO: This the Lightning implementation, which doesn't work for Qrack yet, below.

        // Update tape caching if required
        // if (tapeRecording) {
        //     this->cache_manager.addOperation("QubitUnitary", {}, dev_wires, inverse, matrix,
        //                                      dev_controlled_wires, controlled_values);
        // }
    }
    auto Expval(ObsIdType id) -> double override
    {
        const QrackObservable& obs = obs_cache[id];
        return qsim->ExpectationPauliAll(obs.wires, obs.obs);
    }
    auto Var(ObsIdType id) -> double override
    {
        const QrackObservable& obs = obs_cache[id];
        return qsim->VariancePauliAll(obs.wires, obs.obs);
    }
    void State(DataView<std::complex<double>, 1>& sv) override
    {
        RT_FAIL_IF(sv.size() != (size_t)qsim->GetMaxQPower(), "Invalid size for the pre-allocated state vector");
#if FPPOW == 6
        qsim->GetQuantumState(&(*(sv.begin())));
#else
        std::unique_ptr<Qrack::complex> _sv(new Qrack::complex[sv.size()]);
        qsim->GetQuantumState(_sv.get());
        std::copy(_sv.get(), _sv.get() + sv.size(), sv.begin());
#endif
    }
    void Probs(DataView<double, 1>& p) override
    {
        RT_FAIL_IF(p.size() != (size_t)qsim->GetMaxQPower(), "Invalid size for the pre-allocated probabilities vector");
#if FPPOW == 6
        qsim->GetProbs(&(*(p.begin())));
#else
        std::unique_ptr<Qrack::real1> _p(new Qrack::real1[p.size()]);
        qsim->GetProbs(_p.get());
        std::copy(_p.get(), _p.get() + p.size(), p.begin());
#endif
    }
    void PartialProbs(DataView<double, 1> &p, const std::vector<QubitIdType> &wires) override
    {
        RT_FAIL_IF((size_t)Qrack::pow2(wires.size()) != p.size(), "Invalid size for the pre-allocated probabilities vector");
        std::vector<bitLenInt> ids(wires.size());
        std::transform(wires.begin(), wires.end(), ids.end(), [](QubitIdType a) { return (bitLenInt)a; });
#if FPPOW == 6
        qsim->ProbBitsAll(ids, &(*(p.begin())));
#else
        std::unique_ptr<Qrack::real1> _p(new Qrack::real1[p.size()]);
        qsim->ProbBitsAll(ids, _p.get());
        std::copy(_p.get(), _p.get() + p.size(), p.begin());
#endif
    }
    void Sample(DataView<double, 2> &samples, size_t shots) override
    {
        // TODO: We could suggest, for upstream, that "shots" is a redundant parameter
        // that could be instead implied by the size of "samples."
        RT_FAIL_IF(samples.size() != shots, "Invalid size for the pre-allocated samples");

        std::vector<bitCapInt> qPowers(qsim->GetQubitCount());
        for (bitLenInt i = 0U; i < qPowers.size(); ++i) {
            qPowers[i] = Qrack::pow2(i);
        }
        auto q_samples = qsim->MultiShotMeasureMask(qPowers, shots);

        auto samplesIter = samples.begin();
        for (size_t shot = 0U; shot < shots; ++shot) {
            bitCapInt sample = q_samples[shot];
            for (size_t wire = 0U; wire < qPowers.size(); ++wire) {
                *(samplesIter++) = bi_to_double((sample >> wire) & 1U);
            }
        }
    }
    void PartialSample(DataView<double, 2> &samples, const std::vector<QubitIdType> &wires, size_t shots) override
    {
        // TODO: We could suggest, for upstream, that "shots" is a redundant parameter
        // that could be instead implied by the size of "samples."
        RT_FAIL_IF(samples.size() != shots, "Invalid size for the pre-allocated samples");

        std::vector<bitCapInt> qPowers(wires.size());
        for (size_t i = 0U; i < qPowers.size(); ++i) {
            qPowers[i] = Qrack::pow2((bitLenInt)wires[i]);
        }
        auto q_samples = qsim->MultiShotMeasureMask(qPowers, shots);

        auto samplesIter = samples.begin();
        for (size_t shot = 0U; shot < shots; ++shot) {
            bitCapInt sample = q_samples[shot];
            for (size_t wire = 0U; wire < qPowers.size(); ++wire) {
                *(samplesIter++) = bi_to_double((sample >> wire) & 1U);
            }
        }
    }
    void Counts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                size_t shots) override
    {
        // TODO: We could suggest, for upstream, that "shots" is a redundant parameter
        // that could be instead implied by the size of "eigvals"/"counts".
        const size_t numQubits = qsim->GetQubitCount();
        const size_t numElements = (size_t)qsim->GetMaxQPower();

        RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
                   "Invalid size for the pre-allocated counts");

        std::vector<bitCapInt> qPowers(numQubits);
        for (bitLenInt i = 0U; i < qPowers.size(); ++i) {
            qPowers[i] = Qrack::pow2(i);
        }
        auto q_samples = qsim->MultiShotMeasureMask(qPowers, shots);

        std::iota(eigvals.begin(), eigvals.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t shot = 0; shot < shots; ++shot) {
            bitCapInt sample = q_samples[shot];
            std::bitset<1U << QBCAPPOW> basisState;
            size_t idx = numQubits;
            for (size_t wire = 0; wire < numQubits; wire++) {
                basisState[--idx] = bi_compare_0((sample >> wire) & 1U);
            }
            ++counts(static_cast<size_t>(basisState.to_ulong()));
        }
    }

    void PartialCounts(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires, size_t shots) override
    {
        // TODO: We could suggest, for upstream, that "shots" is a redundant parameter
        // that could be instead implied by the size of "eigvals"/"counts".
        const size_t numQubits = wires.size();
        const size_t numElements = (size_t)Qrack::pow2(numQubits);

        RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
                   "Invalid size for the pre-allocated counts");

        std::vector<bitCapInt> qPowers(wires.size());
        for (size_t i = 0U; i < qPowers.size(); ++i) {
            qPowers[i] = Qrack::pow2(wires[i]);
        }
        auto q_samples = qsim->MultiShotMeasureMask(qPowers, shots);

        std::iota(eigvals.begin(), eigvals.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t shot = 0; shot < shots; ++shot) {
            bitCapInt sample = q_samples[shot];
            std::bitset<1U << QBCAPPOW> basisState;
            size_t idx = numQubits;
            for (size_t wire = 0; wire < numQubits; wire++) {
                basisState[--idx] = bi_compare_0((sample >> wire) & 1U);
            }
            ++counts(static_cast<size_t>(basisState.to_ulong()));
        }
    }

    void Gradient(std::vector<DataView<double, 1>> &, const std::vector<size_t> &) override {}
};

GENERATE_DEVICE_FACTORY(QrackDevice, QrackDevice);
