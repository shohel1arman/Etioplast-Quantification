// Colours must match CLASS_DEFINITIONS in mainserver/myapp/config.py.
// The backend values are OpenCV BGR tuples, so (255, 0, 0) is drawn blue.
export function Legend() {
  const items = [
    { label: "Etioplast", color: "#0000ff" },     // BGR (255, 0, 0)
    { label: "PLB", color: "#00ff00" },           // BGR (0, 255, 0)
    { label: "Prothylakoid", color: "#ff0000" },  // BGR (0, 0, 255)
    { label: "Plastoglobule", color: "#00ffff" }, // BGR (255, 255, 0)
    { label: "Starch Grain", color: "#ff00ff" },  // BGR (255, 0, 255)
  ];

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-200/50 p-4 shadow-md">
      <h3 className="text-slate-800 font-bold mb-4">Structure Color Legend</h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        {items.map((item) => (
          <div key={item.label} className="flex items-center space-x-2">
            <span
              className="inline-block w-3 h-3 rounded-full border border-slate-300"
              style={{ backgroundColor: item.color }}
            ></span>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
