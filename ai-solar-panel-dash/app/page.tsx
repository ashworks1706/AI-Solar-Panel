// pages/index.js
import { useState, useEffect } from "react";
import {
  Navbar,
  Button,
  Card,
  CardBody,
  CardHeader,
  CardFooter,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Spinner,
  Tooltip,
  Progress,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Switch,
} from "@nextui-org/react";
import {
  PlayCircleIcon,
  PauseCircleIcon,
  CameraIcon,
  CloudIcon,
  SunIcon,
  ArrowPathIcon,
  AdjustmentsHorizontalIcon,
  ChartBarIcon,
  ExclamationCircleIcon,
  ChevronDownIcon,
} from "@heroicons/react/24/outline";

import { db } from "../firebase/config";

import { collection, getDocs, query, orderBy, limit } from "firebase/firestore";
import axios from "axios";

// API base URL - replace with your Flask server address
const API_BASE_URL = "http://localhost:5000";

// Define the ModelLog type
type ModelLog = {
  id: string;
  timestamp: any;
  model_details?: {
    detections?: any[];
  };
  raspberry_details?: {
    cpu_percent?: number;
    memory_percent?: number;
  };
  image_url?: string;
};

export default function Home() {
  // State for system status
  const [systemStatus, setSystemStatus] = useState({
    camera_active: false,
    interval_time: 60,
    next_interval_time: null,
    last_detection_time: null,
    model_loaded: false,
    weather_data: null as any,
  });

  // State for model logs
  const [modelLogs, setModelLogs] = useState<any>([]);
  const [programLogs, setProgramLogs] = useState<any>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("status");
  const [customInterval, setCustomInterval] = useState(60);

  // Fetch system status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/status`);
        setSystemStatus(response.data);
      } catch (error) {
        console.error("Error fetching system status:", error);
      }
    };

    // Fetch status every 5 seconds
    fetchStatus();
    const intervalId = setInterval(fetchStatus, 5000);

    return () => clearInterval(intervalId);
  }, []);

  // Fetch model logs from Firebase
  useEffect(() => {
    const fetchModelLogs = async () => {
      try {
        const modelLogsQuery = query(
          collection(db, "ModelLog"),
          orderBy("timestamp", "desc"),
          limit(5)
        );

        const programLogsQuery = query(
          collection(db, "ProgramLog"),
          orderBy("timestamp", "desc"),
          limit(5)
        );

        const modelLogsSnapshot = await getDocs(modelLogsQuery);
        const programLogsSnapshot = await getDocs(programLogsQuery);

        const modelLogsData = modelLogsSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));

        const programLogsData = programLogsSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));

        setModelLogs(modelLogsData);
        setProgramLogs(programLogsData);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching logs:", error);
        setLoading(false);
      }
    };

    fetchModelLogs();
    // Fetch logs every 10 seconds
    const logsInterval = setInterval(fetchModelLogs, 10000);

    return () => clearInterval(logsInterval);
  }, []);

  // Toggle camera status
  const toggleCamera = async () => {
    try {
      const action = systemStatus.camera_active ? "stop" : "start";
      await axios.put(`${API_BASE_URL}/start_stop_camera`, { action });

      // Update local state immediately for better UX
      setSystemStatus((prev) => ({
        ...prev,
        camera_active: !prev.camera_active,
      }));
    } catch (error) {
      console.error("Error toggling camera:", error);
    }
  };

  // Change interval time
  const changeInterval = async (interval: number) => {
    try {
      await axios.put(`${API_BASE_URL}/change_interval`, { interval });

      // Update local state
      setSystemStatus((prev) => ({
        ...prev,
        interval_time: interval,
      }));
    } catch (error) {
      console.error("Error changing interval:", error);
    }
  };

  // Format timestamp for display
  const formatTimestamp = (timestamp: any) => {
    if (!timestamp) return "N/A";

    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      {/* Header */}
      <Navbar className="bg-blue-800 text-white">
        <div className="flex justify-between items-center w-full">
          <div className="flex items-center">
            <SunIcon className="h-8 w-8 mr-2" />
            <p className="font-bold text-xl">AI Solar Panel Dashboard</p>
          </div>
          <div className="flex items-center">
            <Tooltip
              content={
                systemStatus.model_loaded ? "Model loaded" : "Model not loaded"
              }
            >
              <div
                className={`h-3 w-3 rounded-full mr-2 ${
                  systemStatus.model_loaded ? "bg-green-500" : "bg-red-500"
                }`}
              ></div>
            </Tooltip>
            <p className="text-sm">{new Date().toLocaleString()}</p>
          </div>
        </div>
      </Navbar>

      <div className="flex flex-1">
        {/* Sidebar */}
        <div className="w-64 bg-blue-900 text-white p-4">
          <div className="space-y-4">
            <div
              className={`p-2 rounded cursor-pointer hover:bg-blue-700 ${
                activeTab === "status" ? "bg-blue-700" : ""
              }`}
              onClick={() => setActiveTab("status")}
            >
              <div className="flex items-center">
                <ChartBarIcon className="h-5 w-5 mr-2" />
                System Status
              </div>
            </div>
            <div
              className={`p-2 rounded cursor-pointer hover:bg-blue-700 ${
                activeTab === "modelLogs" ? "bg-blue-700" : ""
              }`}
              onClick={() => setActiveTab("modelLogs")}
            >
              <div className="flex items-center">
                <CameraIcon className="h-5 w-5 mr-2" />
                Model Logs
              </div>
            </div>
            <div
              className={`p-2 rounded cursor-pointer hover:bg-blue-700 ${
                activeTab === "programLogs" ? "bg-blue-700" : ""
              }`}
              onClick={() => setActiveTab("programLogs")}
            >
              <div className="flex items-center">
                <AdjustmentsHorizontalIcon className="h-5 w-5 mr-2" />
                Program Logs
              </div>
            </div>
            <div
              className={`p-2 rounded cursor-pointer hover:bg-blue-700 ${
                activeTab === "weather" ? "bg-blue-700" : ""
              }`}
              onClick={() => setActiveTab("weather")}
            >
              <div className="flex items-center">
                <CloudIcon className="h-5 w-5 mr-2" />
                Weather Data
              </div>
            </div>
            <div
              className={`p-2 rounded cursor-pointer hover:bg-blue-700 ${
                activeTab === "controls" ? "bg-blue-700" : ""
              }`}
              onClick={() => setActiveTab("controls")}
            >
              <div className="flex items-center">
                <AdjustmentsHorizontalIcon className="h-5 w-5 mr-2" />
                System Controls
              </div>
            </div>
          </div>

          <div className="mt-10">
            <Card className="bg-blue-800">
              <CardBody>
                <p className="text-sm">System Info</p>
                <div className="mt-2 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Camera:</span>
                    <span
                      className={
                        systemStatus.camera_active
                          ? "text-green-400"
                          : "text-red-400"
                      }
                    >
                      {systemStatus.camera_active ? "Active" : "Inactive"}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Interval:</span>
                    <span>{systemStatus.interval_time}s</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span>Model:</span>
                    <span>
                      {systemStatus.model_loaded ? "Loaded" : "Not Loaded"}
                    </span>
                  </div>
                </div>
              </CardBody>
            </Card>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6">
          {loading ? (
            <div className="flex justify-center items-center h-full">
              <Spinner size="lg" color="primary" />
            </div>
          ) : (
            <>
              {/* Status Dashboard */}
              {activeTab === "status" && (
                <div>
                  <h1 className="text-2xl font-bold mb-6">System Dashboard</h1>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <Card>
                      <CardBody>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-500">
                              Camera Status
                            </p>
                            <p className="text-2xl font-bold">
                              {systemStatus.camera_active
                                ? "Active"
                                : "Inactive"}
                            </p>
                          </div>
                          <div
                            className={`rounded-full p-3 ${
                              systemStatus.camera_active
                                ? "bg-green-100"
                                : "bg-red-100"
                            }`}
                          >
                            <CameraIcon
                              className={`h-6 w-6 ${
                                systemStatus.camera_active
                                  ? "text-green-600"
                                  : "text-red-600"
                              }`}
                            />
                          </div>
                        </div>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardBody>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-500">
                              Current Interval
                            </p>
                            <p className="text-2xl font-bold">
                              {systemStatus.interval_time}s
                            </p>
                          </div>
                          <div className="rounded-full p-3 bg-blue-100">
                            <ArrowPathIcon className="h-6 w-6 text-blue-600" />
                          </div>
                        </div>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardBody>
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-gray-500">Weather</p>
                            <p className="text-2xl font-bold">
                              {systemStatus.weather_data?.weather_condition ||
                                "Unknown"}
                            </p>
                          </div>
                          <div className="rounded-full p-3 bg-yellow-100">
                            <CloudIcon className="h-6 w-6 text-yellow-600" />
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                        <h4 className="font-bold text-large">
                          Recent Detections
                        </h4>
                        <p className="text-tiny text-default-500">
                          Last 5 sun detections
                        </p>
                      </CardHeader>
                      <CardBody>
                        {modelLogs.length > 0 ? (
                          <Table aria-label="Recent detections">
                            <TableHeader>
                              <TableColumn>TIME</TableColumn>
                              <TableColumn>DETECTIONS</TableColumn>
                              <TableColumn>OFFSET</TableColumn>
                            </TableHeader>
                            <TableBody>
                              {modelLogs.map((log: any) => (
                                <TableRow key={log.id}>
                                  <TableCell>
                                    {formatTimestamp(log.timestamp)}
                                  </TableCell>
                                  <TableCell>
                                    {log.model_details?.detections?.length || 0}
                                  </TableCell>
                                  <TableCell>
                                    {log.model_details?.detections?.[0] ? (
                                      <span>
                                        X:{" "}
                                        {log.model_details.detections[0].distance_x.toFixed(
                                          1
                                        )}
                                        , Y:{" "}
                                        {log.model_details.detections[0].distance_y.toFixed(
                                          1
                                        )}
                                      </span>
                                    ) : (
                                      "None"
                                    )}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        ) : (
                          <p className="text-center py-4 text-gray-500">
                            No detection logs found
                          </p>
                        )}
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                        <h4 className="font-bold text-large">
                          System Controls
                        </h4>
                        <p className="text-tiny text-default-500">
                          Manage camera and detection
                        </p>
                      </CardHeader>
                      <CardBody>
                        <div className="space-y-4">
                          <div className="flex justify-between items-center">
                            <span>Camera Status</span>
                            <Switch
                              isSelected={systemStatus.camera_active}
                              onValueChange={toggleCamera}
                              color="success"
                            />
                          </div>

                          <div className="space-y-2">
                            <p className="text-sm">Change Interval</p>
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={() => changeInterval(60)}
                              >
                                1 min
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => changeInterval(180)}
                              >
                                3 min
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => changeInterval(300)}
                              >
                                5 min
                              </Button>
                              <input
                                type="number"
                                className="w-20 p-2 border rounded"
                                value={customInterval}
                                onChange={(e) =>
                                  setCustomInterval(parseInt(e.target.value))
                                }
                                min="10"
                                max="3600"
                              />
                              <Button
                                size="sm"
                                onClick={() => changeInterval(customInterval)}
                              >
                                Set
                              </Button>
                            </div>
                          </div>

                          <div className="pt-2">
                            <Button
                              color={
                                systemStatus.camera_active
                                  ? "danger"
                                  : "success"
                              }
                              startContent={
                                systemStatus.camera_active ? (
                                  <PauseCircleIcon className="h-5 w-5" />
                                ) : (
                                  <PlayCircleIcon className="h-5 w-5" />
                                )
                              }
                              className="w-full"
                              onClick={toggleCamera}
                            >
                              {systemStatus.camera_active
                                ? "Stop Camera"
                                : "Start Camera"}
                            </Button>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                  </div>
                </div>
              )}

              {/* Model Logs */}
              {activeTab === "modelLogs" && (
                <div>
                  <h1 className="text-2xl font-bold mb-6">Model Logs</h1>
                  <Card>
                    <CardBody>
                      {modelLogs.length > 0 ? (
                        <Table aria-label="Model logs">
                          <TableHeader>
                            <TableColumn>TIME</TableColumn>
                            <TableColumn>DETECTIONS</TableColumn>
                            <TableColumn>CPU USAGE</TableColumn>
                            <TableColumn>MEMORY</TableColumn>
                            <TableColumn>IMAGE</TableColumn>
                          </TableHeader>
                          <TableBody>
                            {modelLogs.map((log: ModelLog) => (
                              <TableRow key={log.id}>
                                <TableCell>
                                  {formatTimestamp(log.timestamp)}
                                </TableCell>
                                <TableCell>
                                  {log.model_details?.detections?.length || 0}
                                </TableCell>
                                <TableCell>
                                  {log.raspberry_details?.cpu_percent || "N/A"}%
                                </TableCell>
                                <TableCell>
                                  {log.raspberry_details?.memory_percent ||
                                    "N/A"}
                                  %
                                </TableCell>
                                <TableCell>
                                  {log.image_url ? (
                                    <Button
                                      size="sm"
                                      as="a"
                                      href={log.image_url}
                                      target="_blank"
                                    >
                                      View
                                    </Button>
                                  ) : (
                                    "N/A"
                                  )}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <p className="text-center py-4 text-gray-500">
                          No model logs found
                        </p>
                      )}
                    </CardBody>
                  </Card>
                </div>
              )}

              {/* Program Logs */}
              {activeTab === "programLogs" && (
                <div>
                  <h1 className="text-2xl font-bold mb-6">Program Logs</h1>
                  <Card>
                    <CardBody>
                      {programLogs.length > 0 ? (
                        <Table aria-label="Program logs">
                          <TableHeader>
                            <TableColumn>TIME</TableColumn>
                            <TableColumn>WEATHER</TableColumn>
                            <TableColumn>INTERVAL FORMULA</TableColumn>
                            <TableColumn>NEXT INTERVAL</TableColumn>
                          </TableHeader>
                          <TableBody>
                            {programLogs.map((log: any) => (
                              <TableRow key={log.id}>
                                <TableCell>
                                  {formatTimestamp(log.timestamp)}
                                </TableCell>
                                <TableCell>
                                  {log.weather_response?.weather_condition ||
                                    "N/A"}
                                </TableCell>
                                <TableCell>
                                  {log.interval_formula || "N/A"}
                                </TableCell>
                                <TableCell>
                                  {formatTimestamp(log.next_interval_time) ||
                                    "N/A"}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <p className="text-center py-4 text-gray-500">
                          No program logs found
                        </p>
                      )}
                    </CardBody>
                  </Card>
                </div>
              )}

              {/* Weather Data */}
              {activeTab === "weather" && (
                <div>
                  <h1 className="text-2xl font-bold mb-6">Weather Data</h1>
                  {systemStatus.weather_data ? (
                    <Card>
                      <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                        <h4 className="font-bold text-large">
                          Current Weather
                        </h4>
                        <p className="text-tiny text-default-500">
                          Last updated:{" "}
                          {formatTimestamp(systemStatus.weather_data.timestamp)}
                        </p>
                      </CardHeader>
                      <CardBody>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-gray-500">Condition</p>
                            <p className="text-xl font-semibold">
                              {systemStatus.weather_data.weather_condition}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">Description</p>
                            <p className="text-xl font-semibold">
                              {systemStatus.weather_data.weather_description}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">Temperature</p>
                            <p className="text-xl font-semibold">
                              {(
                                systemStatus.weather_data.temperature - 273.15
                              ).toFixed(1)}
                              °C
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">Cloud Coverage</p>
                            <p className="text-xl font-semibold">
                              {systemStatus.weather_data.clouds}%
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-500">Wind Speed</p>
                            <p className="text-xl font-semibold">
                              {systemStatus.weather_data.wind_speed} m/s
                            </p>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                  ) : (
                    <Card>
                      <CardBody>
                        <p className="text-center py-4 text-gray-500">
                          No weather data available
                        </p>
                      </CardBody>
                    </Card>
                  )}
                </div>
              )}

              {/* System Controls */}
              {activeTab === "controls" && (
                <div>
                  <h1 className="text-2xl font-bold mb-6">System Controls</h1>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                        <h4 className="font-bold text-large">Camera Control</h4>
                        <p className="text-tiny text-default-500">
                          Manage camera operation
                        </p>
                      </CardHeader>
                      <CardBody className="py-5">
                        <div className="space-y-6">
                          <div className="flex justify-between items-center">
                            <span>Camera Status</span>
                            <Switch
                              isSelected={systemStatus.camera_active}
                              onValueChange={toggleCamera}
                              color="success"
                            />
                          </div>

                          <Button
                            color={
                              systemStatus.camera_active ? "danger" : "success"
                            }
                            startContent={
                              systemStatus.camera_active ? (
                                <PauseCircleIcon className="h-5 w-5" />
                              ) : (
                                <PlayCircleIcon className="h-5 w-5" />
                              )
                            }
                            className="w-full"
                            onClick={toggleCamera}
                          >
                            {systemStatus.camera_active
                              ? "Stop Camera"
                              : "Start Camera"}
                          </Button>

                          <p className="text-sm mt-2">
                            {systemStatus.camera_active
                              ? "Camera is running. Click to stop."
                              : "Camera is stopped. Click to start."}
                          </p>
                        </div>
                      </CardBody>
                    </Card>

                    <Card>
                      <CardHeader className="pb-0 pt-4 px-4 flex-col items-start">
                        <h4 className="font-bold text-large">
                          Interval Settings
                        </h4>
                        <p className="text-tiny text-default-500">
                          Configure detection frequency
                        </p>
                      </CardHeader>
                      <CardBody className="py-5">
                        <div className="space-y-4">
                          <div>
                            <p className="mb-2">
                              Current interval:{" "}
                              <span className="font-bold">
                                {systemStatus.interval_time}s
                              </span>
                            </p>
                            <p className="text-xs text-gray-500">
                              Next detection:{" "}
                              {formatTimestamp(systemStatus.next_interval_time)}
                            </p>
                          </div>

                          <div className="space-y-2">
                            <p className="text-sm">Preset Intervals</p>
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={() => changeInterval(60)}
                              >
                                1 min
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => changeInterval(180)}
                              >
                                3 min
                              </Button>
                              <Button
                                size="sm"
                                onClick={() => changeInterval(300)}
                              >
                                5 min
                              </Button>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <p className="text-sm">Custom Interval (seconds)</p>
                            <div className="flex gap-2">
                              <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={customInterval}
                                onChange={(e) =>
                                  setCustomInterval(parseInt(e.target.value))
                                }
                                min="10"
                                max="3600"
                              />
                              <Button
                                onClick={() => changeInterval(customInterval)}
                              >
                                Set
                              </Button>
                            </div>
                          </div>
                        </div>
                      </CardBody>
                    </Card>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
